#include "SavingsKernel.cuh"
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/copy.h>
#include <cstdint>
#include <cmath>
#include <iostream>
#include <exception>

#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        std::cerr << "[C&W GPU] CUDA error at " << __FILE__ << ":" << __LINE__ << ": " << cudaGetErrorString(err) << std::endl; \
        return false; \
    } \
} while(0)

#define CUDA_CHECK_LAST() do { \
    cudaError_t err = cudaGetLastError(); \
    if (err != cudaSuccess) { \
        std::cerr << "[C&W GPU] CUDA error (last) at " << __FILE__ << ":" << __LINE__ << ": " << cudaGetErrorString(err) << std::endl; \
        return false; \
    } \
} while(0)

namespace cobra {

// ------------------------------------------------------------------
// Kernel: count valid neighbours (j > i) for each customer
// ------------------------------------------------------------------
__global__ void countValidNeighborsKernel(
    const int* __restrict__ neighbors,
    int N,
    int maxNeighbors,
    int* __restrict__ counts) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    const int* nbrs = neighbors + idx * maxNeighbors;
    int cnt = 0;
    for (int pos = 0; pos < maxNeighbors; ++pos) {
        int j = nbrs[pos];
        if (j > idx) ++cnt;
    }
    counts[idx] = cnt;
}

// ------------------------------------------------------------------
// Kernel: fill keys and values for valid (i,j) pairs
// ------------------------------------------------------------------
__global__ void fillSavingsKernel(
    const float* __restrict__ x,
    const float* __restrict__ y,
    const int* __restrict__ neighbors,
    int N,
    int maxNeighbors,
    const int* __restrict__ offsets,
    float lambda,
    float* __restrict__ keys,
    uint64_t* __restrict__ values) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    const int* nbrs = neighbors + idx * maxNeighbors;
    int off = offsets[idx];
    int pos = 0;
    float xi = x[idx], yi = y[idx];
    float d0i = roundf(sqrtf(xi*xi + yi*yi));

    for (int slot = 0; slot < maxNeighbors; ++slot) {
        int j = nbrs[slot];
        if (j > idx) {
            float xj = x[j], yj = y[j];
            float d0j = roundf(sqrtf(xj*xj + yj*yj));
            float dij = roundf(sqrtf((xi-xj)*(xi-xj) + (yi-yj)*(yi-yj)));
            float sav = d0i + d0j - lambda * dij;
            keys[off + pos] = -sav;
            values[off + pos] = (uint64_t)idx << 32 | (uint32_t)j;
            ++pos;
        }
    }
}

// ------------------------------------------------------------------
// Host wrapper: compute and sort savings using GPU (with CPU prefix sum)
// ------------------------------------------------------------------
bool computeSavingsGPU(const Instance& instance,
                       int neighbors_num,
                       double lambda,
                       std::vector<Saving>& savings) {

    std::cout << "[C&W GPU] Entering computeSavingsGPU, N=" << instance.get_customers_num()
              << ", neighbors_num=" << neighbors_num << "\n";

    try {
        const int N = instance.get_customers_num();
        if (N <= 0) return true;

        // --- 1. Prepare host data ---
        const int depot = instance.get_depot();
        const auto& xcoords = instance.get_xcoords();
        const auto& ycoords = instance.get_ycoords();
        double depot_x = xcoords[depot];
        double depot_y = ycoords[depot];

        std::vector<float> h_x(N), h_y(N);
        for (int i = 0; i < N; ++i) {
            int cust = instance.get_customers_begin() + i;
            h_x[i] = (float)(xcoords[cust] - depot_x);
            h_y[i] = (float)(ycoords[cust] - depot_y);
        }

        int maxNeighbors = neighbors_num;
        std::vector<int> h_neighbors(N * maxNeighbors, -1);
        for (int i = 0; i < N; ++i) {
            int cust = instance.get_customers_begin() + i;
            const auto& nbrs = instance.get_neighbors_of(cust);
            int copyCount = std::min((int)nbrs.size(), maxNeighbors);
            for (int p = 0; p < copyCount; ++p) {
                h_neighbors[i * maxNeighbors + p] = nbrs[p] - instance.get_customers_begin();
            }
        }

        // --- 2. Copy to device ---
        thrust::device_vector<float> d_x(h_x);
        thrust::device_vector<float> d_y(h_y);
        thrust::device_vector<int> d_neighbors(h_neighbors);
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK_LAST();

        // --- 3. Count valid neighbours ---
        thrust::device_vector<int> d_counts(N);
        int blockSize = 256;
        int gridSize = (N + blockSize - 1) / blockSize;
        countValidNeighborsKernel<<<gridSize, blockSize>>>(
            thrust::raw_pointer_cast(d_neighbors.data()),
            N, maxNeighbors,
            thrust::raw_pointer_cast(d_counts.data()));
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK_LAST();

        // --- 4. Copy counts to host and compute prefix sum on CPU ---
        std::vector<int> h_counts(N);
        thrust::copy(d_counts.begin(), d_counts.end(), h_counts.begin());
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK_LAST();

        // Compute prefix sum (exclusive) on CPU
        std::vector<int> h_offsets(N + 1);
        h_offsets[0] = 0;
        for (int i = 0; i < N; ++i) {
            h_offsets[i + 1] = h_offsets[i] + h_counts[i];
        }
        int total = h_offsets[N];
        std::cout << "[C&W GPU] Total savings: " << total << std::endl;

        if (total == 0) {
            std::cout << "[C&W GPU] No valid savings pairs found.\n";
            return false;
        }

        // --- 5. Copy offsets back to device ---
        thrust::device_vector<int> d_offsets(N + 1);
        thrust::copy(h_offsets.begin(), h_offsets.end(), d_offsets.begin());
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK_LAST();

        // --- 6. Allocate keys and values ---
        thrust::device_vector<float> d_keys(total);
        thrust::device_vector<uint64_t> d_values(total);
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK_LAST();

        // --- 7. Fill keys and values ---
        float lambda_float = (float)lambda;
        fillSavingsKernel<<<gridSize, blockSize>>>(
            thrust::raw_pointer_cast(d_x.data()),
            thrust::raw_pointer_cast(d_y.data()),
            thrust::raw_pointer_cast(d_neighbors.data()),
            N, maxNeighbors,
            thrust::raw_pointer_cast(d_offsets.data()),
            lambda_float,
            thrust::raw_pointer_cast(d_keys.data()),
            thrust::raw_pointer_cast(d_values.data()));
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK_LAST();

        // --- 8. Sort by key ascending (radix sort) ---
        thrust::sort_by_key(thrust::device, d_keys.begin(), d_keys.end(), d_values.begin());
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK_LAST();

        // --- 9. Single transfer back to host ---
        std::vector<float> h_keys(total);
        std::vector<uint64_t> h_values(total);
        thrust::copy(d_keys.begin(), d_keys.end(), h_keys.begin());
        thrust::copy(d_values.begin(), d_values.end(), h_values.begin());
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK_LAST();

        // --- 10. Convert to Saving structs ---
        savings.clear();
        savings.reserve(total);
        for (int idx = 0; idx < total; ++idx) {
            float sav = -h_keys[idx];
            uint64_t packed = h_values[idx];
            int i = (int)(packed >> 32);
            int j = (int)(packed & 0xFFFFFFFF);
            int orig_i = instance.get_customers_begin() + i;
            int orig_j = instance.get_customers_begin() + j;
            savings.push_back({orig_i, orig_j, (double)sav});
        }

        std::cout << "[C&W GPU] Success! Total savings: " << savings.size() << "\n";
        return true;

    } catch (const std::exception& e) {
        std::cerr << "[C&W GPU] Exception: " << e.what() << std::endl;
        return false;
    } catch (...) {
        std::cerr << "[C&W GPU] Unknown exception.\n";
        return false;
    }
}

} // namespace cobra
