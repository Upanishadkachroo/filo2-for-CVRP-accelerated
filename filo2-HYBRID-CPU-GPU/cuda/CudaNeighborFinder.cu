#include "CudaNeighborFinder.hpp"
#include <cuda_runtime.h>
#include <iostream>
#include <chrono>
#include <algorithm>
#include <cstring>

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
                // swap with parent
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
        // Convert max‑heap to sorted ascending order (by distance)
        // Simple selection sort because K is small (<= 1500)
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

// Kernel: each thread handles one query point. Scans the entire dataset in tiles,
// computing squared distances and maintaining a max‑heap of the k closest.
template <int K, int BLOCK_SIZE, int TILE_SIZE>
__global__ void batchedKNN(const double* __restrict__ x, const double* __restrict__ y, int n,
                           const int* __restrict__ query_ids, int num_queries,
                           int* __restrict__ out_indices, float* __restrict__ out_dists) {
    extern __shared__ float s_tile[]; // shared memory for one tile of dataset (x and y)
    double* s_x = reinterpret_cast<double*>(s_tile);
    double* s_y = s_x + TILE_SIZE;

    int tid = threadIdx.x;
    int query_idx = blockIdx.x * blockDim.x + tid;
    bool active = query_idx < num_queries;
    int qid = active ? query_ids[query_idx] : 0; // actual vertex index

    // Per‑thread heap (stored in registers / local memory)
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
                // pad with zeros (won't be used)
                s_x[i] = 0.0;
                s_y[i] = 0.0;
            }
        }
        __syncthreads();

        if (active) {
            double qx = x[qid];   // query point coordinates
            double qy = y[qid];
            // Compute distances for all points in the tile
            for (int i = 0; i < TILE_SIZE && (tile_start + i) < n; ++i) {
                double dx = qx - s_x[i];
                double dy = qy - s_y[i];
                float dist = static_cast<float>(dx*dx + dy*dy);
                int global_idx = tile_start + i;
                // Exclude the query point itself (distance 0 should not be added)
                if (global_idx != qid) {
                    heap.push(global_idx, dist);
                }
            }
        }
        __syncthreads();
    }

    // Write results to global memory
    if (active) {
        heap.sort(); // now ascending order
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
    // Allocate host result arrays: n * k integers
    std::vector<int> h_indices(n * k);
    std::vector<float> h_dists(n * k);

    // Prepare query_ids (all vertices)
    thrust::device_vector<int> d_query_ids(n);
    thrust::sequence(d_query_ids.begin(), d_query_ids.end());

    // Allocate device output arrays
    int* d_indices;
    float* d_dists;
    cudaMalloc(&d_indices, n * k * sizeof(int));
    cudaMalloc(&d_dists,  n * k * sizeof(float));

    // Parameters
    const int BLOCK_SIZE = 256;
    const int TILE_SIZE = 1024;   // tile of points loaded into shared memory
    int num_blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;

    // Shared memory size: two arrays of double of size TILE_SIZE
    size_t shared_mem = 2 * TILE_SIZE * sizeof(double);

    auto start = std::chrono::high_resolution_clock::now();

    // Dispatch kernel with template parameters (k, block size, tile size)
    if (k == 1500) {
        batchedKNN<1500, 256, 1024><<<num_blocks, BLOCK_SIZE, shared_mem>>>(
            d_x, d_y, n,
            thrust::raw_pointer_cast(d_query_ids.data()), n,
            d_indices, d_dists);
    } else {
        // For other k, we would need to generate different template instances.
        // For simplicity, we only support k=1500. In practice, k is fixed.
        std::cerr << "k=" << k << " not supported in this kernel (only 1500)." << std::endl;
        cudaFree(d_indices); cudaFree(d_dists);
        return {};
    }

    cudaDeviceSynchronize();
    auto end = std::chrono::high_resolution_clock::now();
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    if (verbose) std::cout << "CUDA k‑NN (batched) took " << ms << " ms" << std::endl;

    // Copy results back
    cudaMemcpy(h_indices.data(), d_indices, n * k * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_dists.data(),   d_dists,  n * k * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_indices);
    cudaFree(d_dists);

    return h_indices;  // flat array, row‑major
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