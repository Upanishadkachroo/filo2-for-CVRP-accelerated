#include "CudaNeighborFinder.hpp"
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/sequence.h>
#include <iostream>
#include <chrono>
#include <algorithm>

namespace cobra {

// Helper: ceiling integer division
static inline int divUp(int a, int b) { return (a + b - 1) / b; }

// ------------------------------------------------------------------
// Kernel: brute‑force k‑NN for a batch of queries
// Template parameter K = number of neighbours (compile‑time constant)
// ------------------------------------------------------------------
template<int K>
__global__ void bruteForceKNN(const float* __restrict__ x,
                              const float* __restrict__ y,
                              int n,
                              const int* __restrict__ query_ids,
                              int num_queries,
                              int* __restrict__ out_indices,
                              float* __restrict__ out_dists) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= num_queries) return;

    int qid = query_ids[tid];
    float qx = x[qid];
    float qy = y[qid];

    // Max‑heap of size K (stored in registers / local memory)
    int heap_idx[K];
    float heap_dist[K];
    int heap_size = 0;

    // Scan all vertices
    for (int j = 0; j < n; ++j) {
        if (j == qid) continue;               // skip self
        float dx = qx - x[j];
        float dy = qy - y[j];
        float dist = dx*dx + dy*dy;

        if (heap_size < K) {
            // Insert at the end and sift up
            heap_idx[heap_size] = j;
            heap_dist[heap_size] = dist;
            int pos = heap_size;
            while (pos > 0 && heap_dist[pos] > heap_dist[(pos-1)/2]) {
                std::swap(heap_idx[pos], heap_idx[(pos-1)/2]);
                std::swap(heap_dist[pos], heap_dist[(pos-1)/2]);
                pos = (pos-1)/2;
            }
            ++heap_size;
        } else if (dist < heap_dist[0]) {
            // Replace root and sift down
            heap_idx[0] = j;
            heap_dist[0] = dist;
            int pos = 0;
            while (true) {
                int left  = 2*pos + 1;
                int right = 2*pos + 2;
                int largest = pos;
                if (left < K && heap_dist[left] > heap_dist[largest]) largest = left;
                if (right < K && heap_dist[right] > heap_dist[largest]) largest = right;
                if (largest == pos) break;
                std::swap(heap_idx[pos], heap_idx[largest]);
                std::swap(heap_dist[pos], heap_dist[largest]);
                pos = largest;
            }
        }
    }

    // Convert max‑heap to ascending order (simple selection sort – K ≤ 1500)
    for (int i = 0; i < heap_size-1; ++i) {
        int min_pos = i;
        for (int j = i+1; j < heap_size; ++j) {
            if (heap_dist[j] < heap_dist[min_pos]) min_pos = j;
        }
        if (min_pos != i) {
            std::swap(heap_idx[i], heap_idx[min_pos]);
            std::swap(heap_dist[i], heap_dist[min_pos]);
        }
    }

    // Write results to global memory
    int base = tid * K;
    for (int i = 0; i < K; ++i) {
        if (i < heap_size) {
            out_indices[base + i] = heap_idx[i];
            out_dists[base + i]   = heap_dist[i];
        } else {
            out_indices[base + i] = -1;
            out_dists[base + i]   = 1e30f;
        }
    }
}

// ------------------------------------------------------------------
// Host implementation
// ------------------------------------------------------------------

CudaNeighborFinder::CudaNeighborFinder(const std::vector<double>& x, const std::vector<double>& y)
    : n(x.size()), h_x(x), h_y(y) {
    // Convert to float (saves GPU memory and increases bandwidth)
    std::vector<float> xf(n), yf(n);
    for (int i = 0; i < n; ++i) {
        xf[i] = static_cast<float>(x[i]);
        yf[i] = static_cast<float>(y[i]);
    }
    cudaMalloc(&d_x, n * sizeof(float));
    cudaMalloc(&d_y, n * sizeof(float));
    cudaMemcpy(d_x, xf.data(), n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, yf.data(), n * sizeof(float), cudaMemcpyHostToDevice);
}

CudaNeighborFinder::~CudaNeighborFinder() {
    if (d_x) cudaFree(d_x);
    if (d_y) cudaFree(d_y);
}

std::vector<int> CudaNeighborFinder::computeAllNeighborsFlat(int k, bool verbose) {
    if (k <= 0 || k > 1500) {
        std::cerr << "CudaNeighborFinder: k must be between 1 and 1500 (got " << k << ")" << std::endl;
        return {};
    }

    // Allocate device output buffers
    int* d_indices = nullptr;
    float* d_dists = nullptr;
    cudaMalloc(&d_indices, n * k * sizeof(int));
    cudaMalloc(&d_dists,   n * k * sizeof(float));

    // Prepare query IDs (all vertices)
    thrust::device_vector<int> d_qids(n);
    thrust::sequence(d_qids.begin(), d_qids.end());

    // Launch configuration: 256 threads per block
    const int BLOCK_SIZE = 256;
    int gridSize = divUp(n, BLOCK_SIZE);

    auto start = std::chrono::high_resolution_clock::now();

    // Dispatch the appropriate kernel specialization
    if (k == 500) {
        bruteForceKNN<500><<<gridSize, BLOCK_SIZE>>>(
            d_x, d_y, n,
            thrust::raw_pointer_cast(d_qids.data()), n,
            d_indices, d_dists);
    } else if (k == 1000) {
        bruteForceKNN<1000><<<gridSize, BLOCK_SIZE>>>(
            d_x, d_y, n,
            thrust::raw_pointer_cast(d_qids.data()), n,
            d_indices, d_dists);
    } else if (k == 1500) {
        bruteForceKNN<1500><<<gridSize, BLOCK_SIZE>>>(
            d_x, d_y, n,
            thrust::raw_pointer_cast(d_qids.data()), n,
            d_indices, d_dists);
    } else {
        std::cerr << "Unsupported k = " << k << ". Please add a kernel specialization for this value." << std::endl;
        cudaFree(d_indices);
        cudaFree(d_dists);
        return {};
    }

    cudaDeviceSynchronize();
    auto end = std::chrono::high_resolution_clock::now();
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    if (verbose) {
        std::cout << "CUDA brute‑force k‑NN (k=" << k << ") took " << ms << " ms" << std::endl;
    }

    // Copy results back to host
    std::vector<int> h_indices(n * k);
    std::vector<float> h_dists(n * k);
    cudaMemcpy(h_indices.data(), d_indices, n * k * sizeof(int),   cudaMemcpyDeviceToHost);
    cudaMemcpy(h_dists.data(),   d_dists,   n * k * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_indices);
    cudaFree(d_dists);

    return h_indices;
}

std::vector<std::vector<int>> CudaNeighborFinder::computeAllNeighbors(int k, bool verbose) {
    auto flat = computeAllNeighborsFlat(k, verbose);
    std::vector<std::vector<int>> result(n);
    for (int i = 0; i < n; ++i) {
        result[i].resize(k);
        memcpy(result[i].data(), flat.data() + i * k, k * sizeof(int));
    }
    return result;
}

} // namespace cobra