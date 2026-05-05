// instance/Instance.cu

#include "Instance.hpp"
#include "../base/gpu_utils.cuh"
#include <cuda_runtime.h>
#include <vector>
#include <cmath>

namespace cobra {

// ---------------------------------------------------------------------------
// Device globals (visible across compilation units via extern if needed)
// ---------------------------------------------------------------------------
float* d_xcoords = nullptr;
float* d_ycoords = nullptr;
int    d_n_vertices = 0;

static constexpr int COST_BLOCK = 256;

__global__ void batched_cost_kernel(
    const float* __restrict__ xc,
    const float* __restrict__ yc,
    const int2*  __restrict__ pairs,
    int*         __restrict__ out,
    int n_pairs)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_pairs) return;

    int i = pairs[tid].x;
    int j = pairs[tid].y;

    float dx = xc[i] - xc[j];
    float dy = yc[i] - yc[j];

    out[tid] = __float2int_rn(sqrtf(dx * dx + dy * dy));
}

// ---------------------------------------------------------------------------
// Upload coordinates to GPU
// ---------------------------------------------------------------------------
void instance_upload_coords(const Instance& inst)
{
    const auto& x = inst.get_xcoords();
    const auto& y = inst.get_ycoords();

    size_t n = x.size();

    if (d_n_vertices != static_cast<int>(n)) {

        if (d_xcoords) cudaFree(d_xcoords);
        if (d_ycoords) cudaFree(d_ycoords);

        CUDA_CHECK(cudaMalloc(&d_xcoords, n * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_ycoords, n * sizeof(float)));

        d_n_vertices = static_cast<int>(n);
    }

    std::vector<float> fx(n), fy(n);
    for (size_t i = 0; i < n; ++i) {
        fx[i] = static_cast<float>(x[i]);
        fy[i] = static_cast<float>(y[i]);
    }

    CUDA_CHECK(cudaMemcpy(d_xcoords, fx.data(), n * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_ycoords, fy.data(), n * sizeof(float), cudaMemcpyHostToDevice));
}

// ---------------------------------------------------------------------------
// Free GPU memory
// ---------------------------------------------------------------------------
void instance_free_coords()
{
    if (d_xcoords) {
        cudaFree(d_xcoords);
        d_xcoords = nullptr;
    }

    if (d_ycoords) {
        cudaFree(d_ycoords);
        d_ycoords = nullptr;
    }

    d_n_vertices = 0;
}

// ---------------------------------------------------------------------------
// Launch kernel
// ---------------------------------------------------------------------------
void launch_batched_cost(
    const int2* d_pairs,
    int*        d_out,
    int         n_pairs,
    cudaStream_t stream)
{
    if (n_pairs == 0) return;

    int grid = (n_pairs + COST_BLOCK - 1) / COST_BLOCK;

    batched_cost_kernel<<<grid, COST_BLOCK, 0, stream>>>(
        d_xcoords, d_ycoords, d_pairs, d_out, n_pairs);

    CUDA_CHECK(cudaGetLastError());
}

} // namespace cobra