#include "../instance/Instance.hpp"
#include "../base/gpu_utils.cuh"

#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/copy.h>
#include <vector>

extern float* d_xcoords;
extern float* d_ycoords;
extern int d_n_vertices;

struct Saving {
    int i;
    int j;
    float value;
};


struct SavingGPU {
    int i;
    int j;
    float value;
};


struct SavingComparator {
    __host__ __device__
    bool operator()(const SavingGPU& a, const SavingGPU& b) const {
        return a.value > b.value;
    }
};


__global__ void savings_kernel(
    const int* __restrict__ d_i,
    const int* __restrict__ d_j,
    SavingGPU* __restrict__ d_savings,
    int n,
    float lambda,
    int depot
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;

    // Shared memory: cache depot coordinates
    // Only one load per block instead of per thread
    __shared__ float depot_x;
    __shared__ float depot_y;

    if (threadIdx.x == 0) {
        depot_x = d_xcoords[depot];
        depot_y = d_ycoords[depot];
    }

    __syncthreads();

    int i = d_i[tid];
    int j = d_j[tid];

    float xi = d_xcoords[i];
    float yi = d_ycoords[i];

    float xj = d_xcoords[j];
    float yj = d_ycoords[j];

    // distances
    float dij = sqrtf((xi - xj)*(xi - xj) + (yi - yj)*(yi - yj));
    float did = sqrtf((xi - depot_x)*(xi - depot_x) +
                      (yi - depot_y)*(yi - depot_y));
    float djd = sqrtf((xj - depot_x)*(xj - depot_x) +
                      (yj - depot_y)*(yj - depot_y));

    float saving = did + djd - lambda * dij;

    d_savings[tid] = {i, j, saving};
}


void compute_savings_gpu(
    const cobra::Instance& instance,
    const std::vector<std::pair<int,int>>& pairs,
    float lambda,
    std::vector<Saving>& savings_out
) {
    int n = static_cast<int>(pairs.size());
    if (n == 0) return;

    std::vector<int> h_i(n);
    std::vector<int> h_j(n);

    for (int k = 0; k < n; ++k) {
        h_i[k] = pairs[k].first;
        h_j[k] = pairs[k].second;
    }

    int* d_i = nullptr;
    int* d_j = nullptr;
    SavingGPU* d_savings_raw = nullptr;

    CUDA_CHECK(cudaMalloc(&d_i, n * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_j, n * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_savings_raw, n * sizeof(SavingGPU)));

    CUDA_CHECK(cudaMemcpy(
        d_i, h_i.data(),
        n * sizeof(int),
        cudaMemcpyHostToDevice
    ));

    CUDA_CHECK(cudaMemcpy(
        d_j, h_j.data(),
        n * sizeof(int),
        cudaMemcpyHostToDevice
    ));


    int grid = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;

    savings_kernel<<<grid, BLOCK_SIZE>>>(
        d_i,
        d_j,
        d_savings_raw,
        n,
        lambda,
        instance.get_depot()
    );

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Thrust sort directly on GPU
    thrust::device_ptr<SavingGPU> dev_ptr(d_savings_raw);

    thrust::sort(
        dev_ptr,
        dev_ptr + n,
        SavingComparator()
    );

    std::vector<SavingGPU> temp(n);

    CUDA_CHECK(cudaMemcpy(
        temp.data(),
        d_savings_raw,
        n * sizeof(SavingGPU),
        cudaMemcpyDeviceToHost
    ));

    savings_out.resize(n);
    for (int k = 0; k < n; ++k) {
        savings_out[k] = {
            temp[k].i,
            temp[k].j,
            temp[k].value
        };
    }

    cudaFree(d_i);
    cudaFree(d_j);
    cudaFree(d_savings_raw);
}

// GPU parallel computation of Clarke & Wright savings.
//
// Each thread computes one saving:
// S(i,j) = d(i, depot) + d(depot, j) - lambda * d(i, j)
//
// Designed to integrate with:
// - Instance.cu (device coordinates)
// - savings.hpp (CPU orchestration)

// Optimized GPU Clarke & Wright savings computation.
//
// Optimizations:
// 1. Shared memory for depot coordinates
// 2. Thrust sorting on GPU
//
// Formula:
// S(i,j) = d(i,depot) + d(depot,j) - lambda*d(i,j)