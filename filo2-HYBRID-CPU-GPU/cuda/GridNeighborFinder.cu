#include "GridNeighborFinder.hpp"
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/scan.h>
#include <thrust/sequence.h>
#include <iostream>
#include <chrono>
#include <cmath>
#include <algorithm>

namespace cobra {

static inline int divUp(int a, int b) { return (a + b - 1) / b; }

// ------------------------------------------------------------
// Kernel 1: count points per cell
// ------------------------------------------------------------
__global__ void countPointsPerCell(const float* x, const float* y, int n,
                                   float min_x, float min_y, float cell_size,
                                   int grid_w, int grid_h, int* cell_counts) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n) {
        int cx = int((x[idx] - min_x) / cell_size);
        int cy = int((y[idx] - min_y) / cell_size);
        cx = max(0, min(cx, grid_w - 1));
        cy = max(0, min(cy, grid_h - 1));
        int cell = cy * grid_w + cx;
        atomicAdd(&cell_counts[cell], 1);
    }
}

// ------------------------------------------------------------
// Kernel 2: fill CSR data arrays
// ------------------------------------------------------------
__global__ void fillCells(const float* x, const float* y, int n,
                          float min_x, float min_y, float cell_size,
                          int grid_w, int grid_h,
                          const int* cell_offsets, int* cell_data, int* cell_counters) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n) {
        int cx = int((x[idx] - min_x) / cell_size);
        int cy = int((y[idx] - min_y) / cell_size);
        cx = max(0, min(cx, grid_w - 1));
        cy = max(0, min(cy, grid_h - 1));
        int cell = cy * grid_w + cx;
        int pos = atomicAdd(&cell_counters[cell], 1);
        cell_data[cell_offsets[cell] + pos] = idx;
    }
}

// ------------------------------------------------------------
// Kernel 3: batched k‑NN using grid expansion (global memory buffers)
// ------------------------------------------------------------
__global__ void batchedGridKNN(const float* __restrict__ x, const float* __restrict__ y,
                               int n,
                               float min_x, float min_y, float cell_size,
                               int grid_w, int grid_h,
                               const int* __restrict__ cell_offsets,
                               const int* __restrict__ cell_data,
                               const int* __restrict__ query_ids, int num_queries,
                               int k,
                               int* __restrict__ out_indices, float* __restrict__ out_dists) {
    int qidx = blockIdx.x;
    if (qidx >= num_queries) return;

    int qid = query_ids[qidx];
    float qx = x[qid];
    float qy = y[qid];

    // Global memory buffers (per block) – not shared, each block has its own copy in global memory.
    // Size is determined at runtime, but we use a fixed maximum.
    const int MAX_CAND = 100000;
    int   cand_idx[MAX_CAND];
    float cand_dist[MAX_CAND];
    int cand_count = 0;

    int cx0 = int((qx - min_x) / cell_size);
    int cy0 = int((qy - min_y) / cell_size);
    cx0 = max(0, min(cx0, grid_w - 1));
    cy0 = max(0, min(cy0, grid_h - 1));

    int radius = 0;
    const int MAX_RADIUS = 50;

    while (cand_count < k && radius <= MAX_RADIUS) {
        int start_x = max(cx0 - radius, 0);
        int end_x   = min(cx0 + radius, grid_w - 1);
        int start_y = max(cy0 - radius, 0);
        int end_y   = min(cy0 + radius, grid_h - 1);
        for (int cy = start_y; cy <= end_y; ++cy) {
            for (int cx = start_x; cx <= end_x; ++cx) {
                int dx = cx - cx0;
                int dy = cy - cy0;
                if (radius > 0 && abs(dx) != radius && abs(dy) != radius) continue;
                int cell = cy * grid_w + cx;
                int start = cell_offsets[cell];
                int end   = cell_offsets[cell + 1];
                for (int p = start; p < end; ++p) {
                    int v = cell_data[p];
                    if (v == qid) continue;
                    if (cand_count >= MAX_CAND) break;
                    float dxv = qx - x[v];
                    float dyv = qy - y[v];
                    cand_dist[cand_count] = dxv*dxv + dyv*dyv;
                    cand_idx[cand_count] = v;
                    ++cand_count;
                }
                if (cand_count >= MAX_CAND) break;
            }
            if (cand_count >= MAX_CAND) break;
        }
        ++radius;
    }

    // Selection sort (partial) to get top k
    int limit = (cand_count < k) ? cand_count : k;
    for (int i = 0; i < limit; ++i) {
        int min_pos = i;
        for (int j = i+1; j < cand_count; ++j) {
            if (cand_dist[j] < cand_dist[min_pos]) min_pos = j;
        }
        if (min_pos != i) {
            float td = cand_dist[i]; cand_dist[i] = cand_dist[min_pos]; cand_dist[min_pos] = td;
            int ti = cand_idx[i]; cand_idx[i] = cand_idx[min_pos]; cand_idx[min_pos] = ti;
        }
    }

    int base = qidx * k;
    for (int i = 0; i < k; ++i) {
        if (i < cand_count) {
            out_indices[base + i] = cand_idx[i];
            out_dists[base + i]   = cand_dist[i];
        } else {
            out_indices[base + i] = -1;
            out_dists[base + i]   = 1e30f;
        }
    }
}

// ------------------------------------------------------------
// Host implementation (unchanged except for using the fixed kernel)
// ------------------------------------------------------------
GridNeighborFinder::GridNeighborFinder(const std::vector<double>& x, const std::vector<double>& y)
    : n(x.size()), h_x(x), h_y(y) {
    std::vector<float> xf(n), yf(n);
    for (int i = 0; i < n; ++i) {
        xf[i] = static_cast<float>(x[i]);
        yf[i] = static_cast<float>(y[i]);
    }
    cudaMalloc(&d_x, n * sizeof(float));
    cudaMalloc(&d_y, n * sizeof(float));
    cudaMemcpy(d_x, xf.data(), n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, yf.data(), n * sizeof(float), cudaMemcpyHostToDevice);

    min_x = *std::min_element(xf.begin(), xf.end());
    max_x = *std::max_element(xf.begin(), xf.end());
    min_y = *std::min_element(yf.begin(), yf.end());
    max_y = *std::max_element(yf.begin(), yf.end());

    float range_x = max_x - min_x;
    float range_y = max_y - min_y;
    const float target_pts_per_cell = 500.0f;
    int target_cells = int(n / target_pts_per_cell) + 1;
    int cells_per_dim = int(sqrtf(target_cells)) + 1;
    cell_size = std::max(range_x / cells_per_dim, range_y / cells_per_dim);
    grid_w = int((range_x + cell_size) / cell_size) + 1;
    grid_h = int((range_y + cell_size) / cell_size) + 1;
    grid_cells = grid_w * grid_h;

    buildGrid();
}

GridNeighborFinder::~GridNeighborFinder() {
    if (d_x) cudaFree(d_x);
    if (d_y) cudaFree(d_y);
    if (d_cell_offsets) cudaFree(d_cell_offsets);
    if (d_cell_data) cudaFree(d_cell_data);
}

void GridNeighborFinder::buildGrid() {
    int* d_cell_counts = nullptr;
    cudaMalloc(&d_cell_counts, grid_cells * sizeof(int));
    cudaMemset(d_cell_counts, 0, grid_cells * sizeof(int));

    int blockSize = 256;
    int gridSize = divUp(n, blockSize);
    countPointsPerCell<<<gridSize, blockSize>>>(d_x, d_y, n, min_x, min_y, cell_size, grid_w, grid_h, d_cell_counts);
    cudaDeviceSynchronize();

    thrust::device_vector<int> d_counts(grid_cells);
    cudaMemcpy(thrust::raw_pointer_cast(d_counts.data()), d_cell_counts, grid_cells * sizeof(int), cudaMemcpyDeviceToDevice);
    thrust::device_vector<int> d_offsets(grid_cells + 1);
    thrust::exclusive_scan(d_counts.begin(), d_counts.end(), d_offsets.begin(), 0);
    cudaMalloc(&d_cell_offsets, (grid_cells + 1) * sizeof(int));
    cudaMemcpy(d_cell_offsets, thrust::raw_pointer_cast(d_offsets.data()), (grid_cells + 1) * sizeof(int), cudaMemcpyDeviceToDevice);

    cudaMalloc(&d_cell_data, n * sizeof(int));

    int* d_counters = nullptr;
    cudaMalloc(&d_counters, grid_cells * sizeof(int));
    cudaMemcpy(d_counters, d_cell_offsets, grid_cells * sizeof(int), cudaMemcpyDeviceToDevice);
    fillCells<<<gridSize, blockSize>>>(d_x, d_y, n, min_x, min_y, cell_size, grid_w, grid_h,
                                       d_cell_offsets, d_cell_data, d_counters);
    cudaDeviceSynchronize();
    cudaFree(d_counters);
    cudaFree(d_cell_counts);
}

std::vector<int> GridNeighborFinder::computeAllNeighborsFlat(int k, bool verbose) {
    int* d_indices = nullptr;
    float* d_dists = nullptr;
    cudaMalloc(&d_indices, (long long)n * k * sizeof(int));
    cudaMalloc(&d_dists,   (long long)n * k * sizeof(float));

    const int BATCH_SIZE = 512;   // queries per kernel launch
    int num_batches = divUp(n, BATCH_SIZE);
    auto start_total = std::chrono::high_resolution_clock::now();

    for (int batch = 0; batch < num_batches; ++batch) {
        int batch_start = batch * BATCH_SIZE;
        int batch_count = std::min(BATCH_SIZE, n - batch_start);

        thrust::device_vector<int> d_qids(batch_count);
        thrust::sequence(d_qids.begin(), d_qids.end(), batch_start);

        int blocks = batch_count;   // one block per query
        batchedGridKNN<<<blocks, 1>>>(
            d_x, d_y, n, min_x, min_y, cell_size, grid_w, grid_h,
            d_cell_offsets, d_cell_data,
            thrust::raw_pointer_cast(d_qids.data()), batch_count, k,
            d_indices + (long long)batch_start * k, d_dists + (long long)batch_start * k);
        cudaDeviceSynchronize();

        if (verbose && (batch % 10 == 0)) {
            std::cout << "Grid batch " << batch << "/" << num_batches << " done.\r" << std::flush;
        }
    }

    std::vector<int> h_indices((long long)n * k);
    std::vector<float> h_dists((long long)n * k);
    cudaMemcpy(h_indices.data(), d_indices, (long long)n * k * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_dists.data(),   d_dists,   (long long)n * k * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_indices);
    cudaFree(d_dists);

    auto end_total = std::chrono::high_resolution_clock::now();
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(end_total - start_total).count();
    if (verbose) std::cout << "\nGrid k‑NN (batched) took " << ms << " ms" << std::endl;

    return h_indices;
}

std::vector<std::vector<int>> GridNeighborFinder::computeAllNeighbors(int k, bool verbose) {
    auto flat = computeAllNeighborsFlat(k, verbose);
    std::vector<std::vector<int>> result(n);
    for (int i = 0; i < n; ++i) {
        result[i].resize(k);
        memcpy(result[i].data(), flat.data() + (long long)i * k, k * sizeof(int));
    }
    return result;
}

} // namespace cobra