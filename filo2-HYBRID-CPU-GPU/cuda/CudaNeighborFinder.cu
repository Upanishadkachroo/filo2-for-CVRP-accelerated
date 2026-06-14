#include "CudaNeighborFinder.hpp"
#include <cuda_runtime.h>
#include <iostream>
#include <chrono>
#include <algorithm>
#include <cstring>
#include <thrust/device_vector.h>
#include <thrust/sequence.h>
#include <thrust/swap.h>

namespace cobra {

// ---------- Fixed‑size heap (stores indices and squared distances) ----------
template <typename T, int K>
struct FixedHeap {
    T indices[K];
    float dists[K];
    int size;

    __device__ FixedHeap() : size(0) {}

    __device__ void push(int idx, float dist) {
        // Insert into heap (max‑heap based on dist)
        if (size < K) {
            indices[size] = idx;
            dists[size] = dist;
            size++;
            // sift up
            int i = size - 1;
            while (i > 0 && dists[i] > dists[(i-1)/2]) {
                int p = (i-1)/2;
                thrust::swap(indices[i], indices[p]);
                thrust::swap(dists[i], dists[p]);
                i = p;
            }
        } else if (dist < dists[0]) {
            // replace root and sift down
            indices[0] = idx;
            dists[0] = dist;
            int i = 0;
            while (true) {
                int left = 2*i + 1;
                int right = 2*i + 2;
                int largest = i;
                if (left < size && dists[left] > dists[largest]) largest = left;
                if (right < size && dists[right] > dists[largest]) largest = right;
                if (largest == i) break;
                thrust::swap(indices[i], indices[largest]);
                thrust::swap(dists[i], dists[largest]);
                i = largest;
            }
        }
    }

    __device__ void sort() {
        // Convert max‑heap to sorted ascending order (simple selection sort)
        for (int i = 0; i < size-1; ++i) {
            int min_idx = i;
            for (int j = i+1; j < size; ++j) {
                if (dists[j] < dists[min_idx]) min_idx = j;
            }
            if (min_idx != i) {
                thrust::swap(indices[i], indices[min_idx]);
                thrust::swap(dists[i], dists[min_idx]);
            }
        }
    }
};

// Kernel constants (tune for your GPU)
constexpr int BLOCK_SIZE = 256;
constexpr int TILE_SIZE = 1024;   // points loaded into shared memory per block

// Kernel: each thread handles one query point.
template <int K>
__global__ void batchedKNN(const double* __restrict__ x, const double* __restrict__ y, int n,
                           const int* __restrict__ query_ids, int num_queries,
                           int* __restrict__ out_indices, float* __restrict__ out_dists) {
    extern __shared__ float s_tile[]; // shared memory for one tile of dataset (x and y)
    double* s_x = reinterpret_cast<double*>(s_tile);
    double* s_y = s_x + TILE_SIZE;

    int tid = threadIdx.x;
    int query_idx = blockIdx.x * blockDim.x + tid;
    bool active = query_idx < num_queries;
    int qid = active ? query_ids[query_idx] : 0;

    FixedHeap<int, K> heap;

    // Process dataset in tiles
    for (int tile_start = 0; tile_start < n; tile_start += TILE_SIZE) {
        // Load tile into shared memory cooperatively
        for (int i = tid; i < TILE_SIZE; i += blockDim.x) {
            int global_idx = tile_start + i;
            if (global_idx < n) {
                s_x[i] = x[global_idx];
                s_y[i] = y[global_idx];
            } else {
                s_x[i] = 0.0;
                s_y[i] = 0.0;
            }
        }
        __syncthreads();

        if (active) {
            double qx = x[qid];
            double qy = y[qid];
            for (int i = 0; i < TILE_SIZE && (tile_start + i) < n; ++i) {
                double dx = qx - s_x[i];
                double dy = qy - s_y[i];
                float dist = static_cast<float>(dx*dx + dy*dy);
                int global_idx = tile_start + i;
                if (global_idx != qid) {
                    heap.push(global_idx, dist);
                }
            }
        }
        __syncthreads();
    }

    // Write results
    if (active) {
        heap.sort();
        int base = query_idx * K;
        for (int i = 0; i < K; ++i) {
            out_indices[base + i] = heap.indices[i];
            out_dists[base + i] = heap.dists[i];
        }
    }
}

// ----------------------------------------------------------------------
// Host implementation
// ----------------------------------------------------------------------

CudaNeighborFinder::CudaNeighborFinder(const std::vector<double>& x, const std::vector<double>& y)
    : n(x.size()), h_x(x), h_y(y) {
    cudaMalloc(&d_x, n * sizeof(double));
    cudaMalloc(&d_y, n * sizeof(double));
    cudaMemcpy(d_x, h_x.data(), n * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, h_y.data(), n * sizeof(double), cudaMemcpyHostToDevice);
}

CudaNeighborFinder::~CudaNeighborFinder() {
    cudaFree(d_x);
    cudaFree(d_y);
}

std::vector<int> CudaNeighborFinder::computeAllNeighborsFlat(int k, bool verbose) {
    const int BATCH_SIZE = 200000;  // vertices per batch (adjust to fit GPU memory)
    int num_batches = (n + BATCH_SIZE - 1) / BATCH_SIZE;

    std::vector<int> h_indices(n * k);
    std::vector<float> h_dists(n * k);

    // Temporary device arrays for one batch
    int* d_batch_indices;
    float* d_batch_dists;
    cudaMalloc(&d_batch_indices, BATCH_SIZE * k * sizeof(int));
    cudaMalloc(&d_batch_dists,   BATCH_SIZE * k * sizeof(float));

    size_t shared_mem = 2 * TILE_SIZE * sizeof(double);  // for s_x and s_y

    auto start = std::chrono::high_resolution_clock::now();

    for (int batch = 0; batch < num_batches; ++batch) {
        int start_idx = batch * BATCH_SIZE;
        int count = std::min(BATCH_SIZE, n - start_idx);

        // Create query_ids for this batch on device
        thrust::device_vector<int> d_query_ids(count);
        thrust::sequence(d_query_ids.begin(), d_query_ids.end(), start_idx);

        // Launch kernel for this batch
        int blocks = (count + BLOCK_SIZE - 1) / BLOCK_SIZE;
        if (k == 1500) {
            batchedKNN<1500><<<blocks, BLOCK_SIZE, shared_mem>>>(
                d_x, d_y, n,
                thrust::raw_pointer_cast(d_query_ids.data()), count,
                d_batch_indices, d_batch_dists);
        } else {
            std::cerr << "k=" << k << " not supported (only 1500)." << std::endl;
            cudaFree(d_batch_indices); cudaFree(d_batch_dists);
            return {};
        }
        cudaDeviceSynchronize();

        // Copy results back to host
        cudaMemcpy(h_indices.data() + start_idx * k, d_batch_indices,
                   count * k * sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_dists.data() + start_idx * k, d_batch_dists,
                   count * k * sizeof(float), cudaMemcpyDeviceToHost);
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    if (verbose) std::cout << "CUDA batched k‑NN took " << ms << " ms" << std::endl;

    cudaFree(d_batch_indices);
    cudaFree(d_batch_dists);
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