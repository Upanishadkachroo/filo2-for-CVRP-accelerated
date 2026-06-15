#include "CudaNeighborFinder.hpp"
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/sequence.h>
#include <iostream>
#include <chrono>

namespace cobra {

template<typename T>
__device__ __forceinline__ void device_swap(T& a, T& b) {
    T tmp = a; a = b; b = tmp;
}

static inline int divUp(int a, int b) { return (a + b - 1) / b; }

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

    int   heap_idx[K];
    float heap_dist[K];
    int   heap_size = 0;

    for (int j = 0; j < n; ++j) {
        if (j == qid) continue;
        float dx   = qx - x[j];
        float dy   = qy - y[j];
        float dist = dx*dx + dy*dy;

        if (heap_size < K) {
            heap_idx[heap_size]  = j;
            heap_dist[heap_size] = dist;
            int pos = heap_size;
            while (pos > 0 && heap_dist[pos] > heap_dist[(pos-1)/2]) {
                device_swap(heap_idx[pos],  heap_idx[(pos-1)/2]);
                device_swap(heap_dist[pos], heap_dist[(pos-1)/2]);
                pos = (pos-1)/2;
            }
            ++heap_size;
        } else if (dist < heap_dist[0]) {
            heap_idx[0]  = j;
            heap_dist[0] = dist;
            int pos = 0;
            while (true) {
                int left    = 2*pos + 1;
                int right   = 2*pos + 2;
                int largest = pos;
                if (left  < K && heap_dist[left]  > heap_dist[largest]) largest = left;
                if (right < K && heap_dist[right] > heap_dist[largest]) largest = right;
                if (largest == pos) break;
                device_swap(heap_idx[pos],  heap_idx[largest]);
                device_swap(heap_dist[pos], heap_dist[largest]);
                pos = largest;
            }
        }
    }

    for (int i = 0; i < heap_size - 1; ++i) {
        int min_pos = i;
        for (int j = i + 1; j < heap_size; ++j)
            if (heap_dist[j] < heap_dist[min_pos]) min_pos = j;
        if (min_pos != i) {
            device_swap(heap_idx[i],  heap_idx[min_pos]);
            device_swap(heap_dist[i], heap_dist[min_pos]);
        }
    }

    int base = tid * K;
    for (int i = 0; i < K; ++i) {
        out_indices[base + i] = (i < heap_size) ? heap_idx[i]  : -1;
        out_dists[base + i]   = (i < heap_size) ? heap_dist[i] : 1e30f;
    }
}

CudaNeighborFinder::CudaNeighborFinder(const std::vector<double>& x,
                                       const std::vector<double>& y)
    : n(static_cast<int>(x.size())), h_x(x), h_y(y) {
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
        std::cerr << "CudaNeighborFinder: k must be 1-1500 (got " << k << ")\n";
        return {};
    }

    int*   d_indices = nullptr;
    float* d_dists   = nullptr;
    cudaMalloc(&d_indices, n * k * sizeof(int));
    cudaMalloc(&d_dists,   n * k * sizeof(float));

    thrust::device_vector<int> d_qids(n);
    thrust::sequence(d_qids.begin(), d_qids.end());
    int* raw_qids = thrust::raw_pointer_cast(d_qids.data());

    const int BLOCK_SIZE = 256;
    int gridSize = divUp(n, BLOCK_SIZE);

    auto start = std::chrono::high_resolution_clock::now();

    if      (k <= 500)  bruteForceKNN<500> <<<gridSize, BLOCK_SIZE>>>(d_x, d_y, n, raw_qids, n, d_indices, d_dists);
    else if (k <= 1000) bruteForceKNN<1000><<<gridSize, BLOCK_SIZE>>>(d_x, d_y, n, raw_qids, n, d_indices, d_dists);
    else                bruteForceKNN<1500><<<gridSize, BLOCK_SIZE>>>(d_x, d_y, n, raw_qids, n, d_indices, d_dists);

    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        std::cerr << "CUDA kernel error: " << cudaGetErrorString(err) << "\n";
        cudaFree(d_indices); cudaFree(d_dists);
        return {};
    }

    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                  std::chrono::high_resolution_clock::now() - start).count();
    if (verbose)
        std::cout << "CUDA brute-force k-NN (k=" << k << ") took " << ms << " ms\n";

    std::vector<int>   h_indices(n * k);
    std::vector<float> h_dists(n * k);
    cudaMemcpy(h_indices.data(), d_indices, n * k * sizeof(int),   cudaMemcpyDeviceToHost);
    cudaMemcpy(h_dists.data(),   d_dists,   n * k * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_indices);
    cudaFree(d_dists);
    return h_indices;
}

std::vector<std::vector<int>> CudaNeighborFinder::computeAllNeighbors(int k, bool verbose) {
    auto flat = computeAllNeighborsFlat(k, verbose);
    if (flat.empty()) return {};
    std::vector<std::vector<int>> result(n);
    for (int i = 0; i < n; ++i) {
        result[i].resize(k);
        memcpy(result[i].data(), flat.data() + i * k, k * sizeof(int));
    }
    return result;
}

} // namespace cobra