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

#define CUDA_CHECK(call) do { \
    cudaError_t err__ = (call); \
    if (err__ != cudaSuccess) { \
        std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ \
                  << " - " << cudaGetErrorString(err__) << std::endl; \
    } \
} while (0)

// ------------------------------------------------------------
// Device swap (host std::swap not usable in kernels)
// ------------------------------------------------------------
template<typename T>
__device__ __forceinline__ void dswap(T& a, T& b) { T t = a; a = b; b = t; }

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
// Kernel 3: per-query top-K via grid expansion using a bounded
// max-heap of size K (templated). No unbounded candidate arrays.
//
// Local memory per thread for K=1500: (4+4)*1500 = 12 KB.
// One thread per query, standard grid/block launch.
// ------------------------------------------------------------
template<int K>
__global__ void batchedGridKNN(const float* __restrict__ x, const float* __restrict__ y,
                               int n,
                               float min_x, float min_y, float cell_size,
                               int grid_w, int grid_h,
                               const int* __restrict__ cell_offsets,
                               const int* __restrict__ cell_data,
                               int query_offset, int num_queries,
                               int k,
                               int* __restrict__ out_indices, float* __restrict__ out_dists) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= num_queries) return;

    int qid = query_offset + tid;
    float qx = x[qid];
    float qy = y[qid];

    // Max-heap of current best k (largest dist at root)
    int   heap_idx[K];
    float heap_dist[K];
    int   heap_size = 0;

    int cx0 = int((qx - min_x) / cell_size);
    int cy0 = int((qy - min_y) / cell_size);
    cx0 = max(0, min(cx0, grid_w - 1));
    cy0 = max(0, min(cy0, grid_h - 1));

    const int MAX_RADIUS = max(grid_w, grid_h);

    for (int radius = 0; radius <= MAX_RADIUS; ++radius) {
        // Early termination: once heap is full, stop if the minimum
        // possible distance to the next ring exceeds current worst-kept dist.
        if (heap_size == k) {
            float min_ring_dist = (float)(radius - 1) * cell_size; // conservative
            if (min_ring_dist > 0.0f && min_ring_dist * min_ring_dist > heap_dist[0]) {
                break;
            }
        }

        int start_x = max(cx0 - radius, 0);
        int end_x   = min(cx0 + radius, grid_w - 1);
        int start_y = max(cy0 - radius, 0);
        int end_y   = min(cy0 + radius, grid_h - 1);

        if (start_x > end_x || start_y > end_y) {
            if (cx0 - radius < 0 && cx0 + radius >= grid_w &&
                cy0 - radius < 0 && cy0 + radius >= grid_h) break;
        }

        for (int cy = start_y; cy <= end_y; ++cy) {
            for (int cx = start_x; cx <= end_x; ++cx) {
                int dx = cx - cx0;
                int dy = cy - cy0;
                // Only process the outer ring (already-visited interior skipped)
                if (radius > 0 && abs(dx) != radius && abs(dy) != radius) continue;

                int cell = cy * grid_w + cx;
                int start = cell_offsets[cell];
                int end   = cell_offsets[cell + 1];
                for (int p = start; p < end; ++p) {
                    int v = cell_data[p];
                    if (v == qid) continue;
                    float dxv = qx - x[v];
                    float dyv = qy - y[v];
                    float dist = dxv*dxv + dyv*dyv;

                    if (heap_size < k) {
                        heap_idx[heap_size]  = v;
                        heap_dist[heap_size] = dist;
                        int pos = heap_size;
                        while (pos > 0 && heap_dist[pos] > heap_dist[(pos-1)/2]) {
                            dswap(heap_idx[pos],  heap_idx[(pos-1)/2]);
                            dswap(heap_dist[pos], heap_dist[(pos-1)/2]);
                            pos = (pos-1)/2;
                        }
                        ++heap_size;
                    } else if (dist < heap_dist[0]) {
                        heap_idx[0]  = v;
                        heap_dist[0] = dist;
                        int pos = 0;
                        while (true) {
                            int l = 2*pos+1, r = 2*pos+2, largest = pos;
                            if (l < k && heap_dist[l] > heap_dist[largest]) largest = l;
                            if (r < k && heap_dist[r] > heap_dist[largest]) largest = r;
                            if (largest == pos) break;
                            dswap(heap_idx[pos],  heap_idx[largest]);
                            dswap(heap_dist[pos], heap_dist[largest]);
                            pos = largest;
                        }
                    }
                }
            }
        }
    }

    // Sort ascending (selection sort over at most k <= K elements)
    for (int i = 0; i < heap_size - 1; ++i) {
        int mp = i;
        for (int j = i+1; j < heap_size; ++j)
            if (heap_dist[j] < heap_dist[mp]) mp = j;
        if (mp != i) {
            dswap(heap_idx[i],  heap_idx[mp]);
            dswap(heap_dist[i], heap_dist[mp]);
        }
    }

    int base = tid * k;
    for (int i = 0; i < k; ++i) {
        if (i < heap_size) {
            out_indices[base + i] = heap_idx[i];
            out_dists[base + i]   = heap_dist[i];
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
    std::vector<float> xf(n), yf(n);
    for (int i = 0; i < n; ++i) {
        xf[i] = static_cast<float>(x[i]);
        yf[i] = static_cast<float>(y[i]);
    }
    CUDA_CHECK(cudaMalloc(&d_x, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_y, n * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_x, xf.data(), n * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_y, yf.data(), n * sizeof(float), cudaMemcpyHostToDevice));

    min_x = *std::min_element(xf.begin(), xf.end());
    max_x = *std::max_element(xf.begin(), xf.end());
    min_y = *std::min_element(yf.begin(), yf.end());
    max_y = *std::max_element(yf.begin(), yf.end());

    float range_x = max_x - min_x;
    float range_y = max_y - min_y;
    const float target_pts_per_cell = 500.0f;
    int target_cells = int(n / target_pts_per_cell) + 1;
    int cells_per_dim = int(sqrtf((float)target_cells)) + 1;
    cell_size = std::max(range_x / cells_per_dim, range_y / cells_per_dim);
    if (cell_size <= 0.0f) cell_size = 1.0f;
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
    CUDA_CHECK(cudaMalloc(&d_cell_counts, grid_cells * sizeof(int)));
    CUDA_CHECK(cudaMemset(d_cell_counts, 0, grid_cells * sizeof(int)));

    int blockSize = 256;
    int gridSize = divUp(n, blockSize);
    countPointsPerCell<<<gridSize, blockSize>>>(d_x, d_y, n, min_x, min_y, cell_size, grid_w, grid_h, d_cell_counts);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    thrust::device_vector<int> d_counts(grid_cells);
    CUDA_CHECK(cudaMemcpy(thrust::raw_pointer_cast(d_counts.data()), d_cell_counts, grid_cells * sizeof(int), cudaMemcpyDeviceToDevice));
    thrust::device_vector<int> d_offsets(grid_cells + 1);
    thrust::exclusive_scan(d_counts.begin(), d_counts.end(), d_offsets.begin(), 0);
    CUDA_CHECK(cudaMalloc(&d_cell_offsets, (grid_cells + 1) * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_cell_offsets, thrust::raw_pointer_cast(d_offsets.data()), (grid_cells + 1) * sizeof(int), cudaMemcpyDeviceToDevice));

    CUDA_CHECK(cudaMalloc(&d_cell_data, n * sizeof(int)));

    int* d_counters = nullptr;
    CUDA_CHECK(cudaMalloc(&d_counters, grid_cells * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_counters, d_cell_offsets, grid_cells * sizeof(int), cudaMemcpyDeviceToDevice));
    fillCells<<<gridSize, blockSize>>>(d_x, d_y, n, min_x, min_y, cell_size, grid_w, grid_h,
                                       d_cell_offsets, d_cell_data, d_counters);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    cudaFree(d_counters);
    cudaFree(d_cell_counts);
}

// ------------------------------------------------------------
// computeAllNeighborsFlat
//
// Chunked: process CHUNK_SIZE queries per kernel launch, copy
// results back to host, free GPU output buffers after the loop.
// Only k <= 1500 is supported (template specialisation K=1500).
// ------------------------------------------------------------
std::vector<int> GridNeighborFinder::computeAllNeighborsFlat(int k, bool verbose) {
    if (k <= 0 || k > 1500) {
        std::cerr << "GridNeighborFinder: k must be 1-1500 (got " << k << ")\n";
        return {};
    }

    // Bound GPU output buffers to ~1.2 GB regardless of N/k
    const long long MAX_OUTPUT_BYTES = 1200LL * 1024 * 1024;
    const int CHUNK_SIZE = static_cast<int>(
        std::max(1LL, std::min((long long)n, MAX_OUTPUT_BYTES / (k * 8LL))));

    int num_chunks = divUp(n, CHUNK_SIZE);
    if (verbose) {
        std::cout << "[GridKNN] N=" << n << " k=" << k
                  << " chunk=" << CHUNK_SIZE << " chunks=" << num_chunks << "\n";
    }

    int*   d_indices = nullptr;
    float* d_dists   = nullptr;
    CUDA_CHECK(cudaMalloc(&d_indices, (long long)CHUNK_SIZE * k * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_dists,   (long long)CHUNK_SIZE * k * sizeof(float)));

    std::vector<int> h_indices((long long)n * k);

    const int BLOCK = 64; // small block: 12KB heap/thread for K=1500 is register/local-mem heavy
    auto start_total = std::chrono::high_resolution_clock::now();

    for (int chunk = 0; chunk < num_chunks; ++chunk) {
        int q_start = chunk * CHUNK_SIZE;
        int q_count = std::min(CHUNK_SIZE, n - q_start);
        int gridSize = divUp(q_count, BLOCK);

        if (k == 1500) {
            batchedGridKNN<1500><<<gridSize, BLOCK>>>(
                d_x, d_y, n, min_x, min_y, cell_size, grid_w, grid_h,
                d_cell_offsets, d_cell_data,
                q_start, q_count, k,
                d_indices, d_dists);
        } else {
            std::cerr << "GridNeighborFinder: only k=1500 is compiled.\n";
            cudaFree(d_indices); cudaFree(d_dists);
            return {};
        }

        cudaError_t err = cudaGetLastError();
        if (err == cudaSuccess) err = cudaDeviceSynchronize();
        if (err != cudaSuccess) {
            std::cerr << "GridNeighborFinder kernel error (chunk " << chunk
                      << "): " << cudaGetErrorString(err) << "\n";
            cudaFree(d_indices); cudaFree(d_dists);
            return {};
        }

        CUDA_CHECK(cudaMemcpy(h_indices.data() + (long long)q_start * k,
                              d_indices,
                              (long long)q_count * k * sizeof(int),
                              cudaMemcpyDeviceToHost));

        if (verbose) {
            std::cout << "Grid chunk " << (chunk+1) << "/" << num_chunks << " done.\r" << std::flush;
        }
    }

    cudaFree(d_indices);
    cudaFree(d_dists);

    auto end_total = std::chrono::high_resolution_clock::now();
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(end_total - start_total).count();
    if (verbose) std::cout << "\nGrid k-NN (chunked) took " << ms << " ms" << std::endl;

    return h_indices;
}

std::vector<std::vector<int>> GridNeighborFinder::computeAllNeighbors(int k, bool verbose) {
    auto flat = computeAllNeighborsFlat(k, verbose);
    if (flat.empty()) return {};
    std::vector<std::vector<int>> result(n);
    for (int i = 0; i < n; ++i) {
        result[i].resize(k);
        memcpy(result[i].data(), flat.data() + (long long)i * k, k * sizeof(int));
    }
    return result;
}

} // namespace cobra