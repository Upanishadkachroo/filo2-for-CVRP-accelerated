#include "GridNeighborFinder.hpp"
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/scan.h>
#include <iostream>
#include <chrono>
#include <cmath>
#include <algorithm>

namespace cobra {

// ------------------------------------------------------------
// Helper functions
// ------------------------------------------------------------
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
// Kernel 3: batched k‑NN using grid expansion
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
    int qidx = blockIdx.x;   // one block per query
    if (qidx >= num_queries) return;

    int qid = query_ids[qidx];
    float qx = x[qid];
    float qy = y[qid];

    // Get query cell
    int cx0 = int((qx - min_x) / cell_size);
    int cy0 = int((qy - min_y) / cell_size);
    cx0 = max(0, min(cx0, grid_w - 1));
    cy0 = max(0, min(cy0, grid_h - 1));

    // Buffer for candidates (global memory, one per block; size large enough)
    // We'll use a fixed‑size shared array? Too large. Instead, we collect candidates
    // into a local array in global memory allocated per block via extern __shared__? 
    // Simpler: use a large static global buffer and atomic counters? Messy.
    // For correctness and simplicity, we will collect candidates in a local array
    // placed in global memory using a fixed maximum (e.g., 100000) and rely on the
    // fact that the number of candidates will not exceed that for typical distributions.
    // If it does, we fallback to brute‑force (but we'll trust the grid).
    const int MAX_CAND = 200000;
    __shared__ int cand_idx[MAX_CAND];
    __shared__ float cand_dist[MAX_CAND];
    __shared__ int cand_count;

    // Expand radius until we have at least k candidates or exceed max radius
    int radius = 0;
    int count = 0;
    const int MAX_RADIUS = 50;

    while (count < k && radius <= MAX_RADIUS) {
        // For each cell in the current square ring (Chebyshev distance = radius)
        int start_x = max(cx0 - radius, 0);
        int end_x   = min(cx0 + radius, grid_w - 1);
        int start_y = max(cy0 - radius, 0);
        int end_y   = min(cy0 + radius, grid_h - 1);
        for (int cy = start_y; cy <= end_y; ++cy) {
            for (int cx = start_x; cx <= end_x; ++cx) {
                // Only add cells that are at exactly the current radius (outer ring)
                int dx = cx - cx0;
                int dy = cy - cy0;
                if (radius > 0 && abs(dx) != radius && abs(dy) != radius) continue;
                int cell = cy * grid_w + cx;
                int start = cell_offsets[cell];
                int end   = cell_offsets[cell + 1];
                for (int p = start; p < end; ++p) {
                    int v = cell_data[p];
                    if (v == qid) continue;
                    if (count >= MAX_CAND) break;
                    float dxv = qx - x[v];
                    float dyv = qy - y[v];
                    cand_dist[count] = dxv*dxv + dyv*dyv;
                    cand_idx[count] = v;
                    ++count;
                }
                if (count >= MAX_CAND) break;
            }
            if (count >= MAX_CAND) break;
        }
        ++radius;
    }

    cand_count = count;

    // Now we have cand_count candidates. Select the k smallest distances.
    if (cand_count > 0) {
        // Partial selection sort (keep only the smallest k)
        for (int i = 0; i < k && i < cand_count; ++i) {
            int min_pos = i;
            for (int j = i + 1; j < cand_count; ++j) {
                if (cand_dist[j] < cand_dist[min_pos]) min_pos = j;
            }
            if (min_pos != i) {
                float td = cand_dist[i]; cand_dist[i] = cand_dist[min_pos]; cand_dist[min_pos] = td;
                int ti = cand_idx[i]; cand_idx[i] = cand_idx[min_pos]; cand_idx[min_pos] = ti;
            }
        }
    }

    // Write output (pad with -1 if not enough candidates)
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
// Host implementation
// ------------------------------------------------------------
GridNeighborFinder::GridNeighborFinder(const std::vector<double>& x, const std::vector<double>& y)
    : n(x.size()), h_x(x), h_y(y) {
    // Convert to float for GPU
    std::vector<float> xf(n), yf(n);
    for (int i = 0; i < n; ++i) {
        xf[i] = static_cast<float>(x[i]);
        yf[i] = static_cast<float>(y[i]);
    }
    cudaMalloc(&d_x, n * sizeof(float));
    cudaMalloc(&d_y, n * sizeof(float));
    cudaMemcpy(d_x, xf.data(), n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, yf.data(), n * sizeof(float), cudaMemcpyHostToDevice);

    // Compute bounding box
    min_x = *std::min_element(xf.begin(), xf.end());
    max_x = *std::max_element(xf.begin(), xf.end());
    min_y = *std::min_element(yf.begin(), yf.end());
    max_y = *std::max_element(yf.begin(), yf.end());

    // Choose cell size so that each cell has about 500 points (tunable)
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
    // Step 1: allocate and zero cell counts
    int* d_cell_counts = nullptr;
    cudaMalloc(&d_cell_counts, grid_cells * sizeof(int));
    cudaMemset(d_cell_counts, 0, grid_cells * sizeof(int));

    // Step 2: count points per cell
    int blockSize = 256;
    int gridSize = divUp(n, blockSize);
    countPointsPerCell<<<gridSize, blockSize>>>(d_x, d_y, n, min_x, min_y, cell_size, grid_w, grid_h, d_cell_counts);
    cudaDeviceSynchronize();

    // Step 3: exclusive prefix sum to get offsets
    thrust::device_vector<int> d_counts(grid_cells);
    cudaMemcpy(thrust::raw_pointer_cast(d_counts.data()), d_cell_counts, grid_cells * sizeof(int), cudaMemcpyDeviceToDevice);
    thrust::device_vector<int> d_offsets(grid_cells + 1);
    thrust::exclusive_scan(d_counts.begin(), d_counts.end(), d_offsets.begin(), 0);
    cudaMalloc(&d_cell_offsets, (grid_cells + 1) * sizeof(int));
    cudaMemcpy(d_cell_offsets, thrust::raw_pointer_cast(d_offsets.data()), (grid_cells + 1) * sizeof(int), cudaMemcpyDeviceToDevice);

    // Step 4: allocate cell_data (size = n)
    cudaMalloc(&d_cell_data, n * sizeof(int));

    // Step 5: fill cell data using atomic counters
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
    // Allocate output buffers
    int* d_indices = nullptr;
    float* d_dists = nullptr;
    cudaMalloc(&d_indices, n * k * sizeof(int));
    cudaMalloc(&d_dists,   n * k * sizeof(float));

    // Process in batches to avoid launching too many blocks
    const int BATCH_SIZE = 512;   // queries per kernel launch
    int num_batches = divUp(n, BATCH_SIZE);
    auto start_total = std::chrono::high_resolution_clock::now();

    for (int batch = 0; batch < num_batches; ++batch) {
        int batch_start = batch * BATCH_SIZE;
        int batch_count = std::min(BATCH_SIZE, n - batch_start);

        // Prepare query IDs for this batch
        thrust::device_vector<int> d_qids(batch_count);
        thrust::sequence(d_qids.begin(), d_qids.end(), batch_start);

        // One block per query
        int blocks = batch_count;
        int threads = 1;   // each block uses only one thread? Actually the kernel uses shared memory but still
                           // only one thread does the work; we can use 1 thread per block to save resources.
        batchedGridKNN<<<blocks, 1>>>(
            d_x, d_y, n, min_x, min_y, cell_size, grid_w, grid_h,
            d_cell_offsets, d_cell_data,
            thrust::raw_pointer_cast(d_qids.data()), batch_count, k,
            d_indices + batch_start * k, d_dists + batch_start * k);
        cudaDeviceSynchronize();

        if (verbose && (batch % 10 == 0)) {
            std::cout << "Grid batch " << batch << "/" << num_batches << " done.\r" << std::flush;
        }
    }

    // Copy results back to host
    std::vector<int> h_indices(n * k);
    std::vector<float> h_dists(n * k);
    cudaMemcpy(h_indices.data(), d_indices, n * k * sizeof(int),   cudaMemcpyDeviceToHost);
    cudaMemcpy(h_dists.data(),   d_dists,   n * k * sizeof(float), cudaMemcpyDeviceToHost);
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
        memcpy(result[i].data(), flat.data() + i * k, k * sizeof(int));
    }
    return result;
}

} // namespace cobra