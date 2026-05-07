#include "instance_gpu.h"
#include "../cpu/Instance.hpp"
#include "../../base/gpu/cuda_utils.h"
#include <cuda_runtime.h>
#include <nvtx3/nvToolsExt.h>      // FIX [4]: NVTX profiling markers
#include <cmath>
#include <cstdio>
#include <cstdlib>

namespace cobra {


float* d_xcoords    = nullptr;
float* d_ycoords    = nullptr;
int    d_n_vertices = 0;

static float* h_pinned_x = nullptr;
static float* h_pinned_y = nullptr;
static int    g_pinned_n = 0;       // capacity of current pinned allocation

// Optimal block size determined at first launch
static int g_cost_block = 256;      // default; overwritten by occupancy query

static_assert(sizeof(float) == 4,
    "float must be 32-bit IEEE-754 for GPU distance rounding to match CPU");

__global__ void batched_cost_kernel(
    const float* __restrict__ xc,
    const float* __restrict__ yc,
    const int2*  __restrict__ pairs,
    int*         __restrict__ out,
    int n_pairs)
{
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_pairs) return;

    const int i = pairs[tid].x;
    const int j = pairs[tid].y;

    const float dx = xc[i] - xc[j];
    const float dy = yc[i] - yc[j];

    // __float2int_rn: rounds to nearest even integer — equivalent to
    // static_cast<int>(val + 0.5) for positive values (fastround contract).
    out[tid] = __float2int_rn(sqrtf(dx * dx + dy * dy));
}

static void ensure_pinned(int n)
{
    if (n <= g_pinned_n) return;           // already big enough

    // Free old buffers if they exist
    if (h_pinned_x) CUDA_CHECK(cudaFreeHost(h_pinned_x));
    if (h_pinned_y) CUDA_CHECK(cudaFreeHost(h_pinned_y));

    CUDA_CHECK(cudaMallocHost(&h_pinned_x, n * sizeof(float)));
    CUDA_CHECK(cudaMallocHost(&h_pinned_y, n * sizeof(float)));
    g_pinned_n = n;
}

void instance_upload_coords(const Instance& inst, cudaStream_t stream)
{
    nvtxRangePush("instance_upload_coords");   
    const auto& x = inst.get_xcoords();
    const auto& y = inst.get_ycoords();
    const int   n = static_cast<int>(x.size());

    // Reallocate device buffers only if size has changed
    if (d_n_vertices != n) {
        if (d_xcoords) { CUDA_CHECK(cudaFree(d_xcoords)); }  
        if (d_ycoords) { CUDA_CHECK(cudaFree(d_ycoords)); }  

        CUDA_CHECK(cudaMalloc(&d_xcoords, n * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_ycoords, n * sizeof(float)));
        d_n_vertices = n;
    }

    // Fill pinned staging buffers (one-time allocation)
    ensure_pinned(n);

    for (int i = 0; i < n; ++i) {
        h_pinned_x[i] = static_cast<float>(x[i]);
        h_pinned_y[i] = static_cast<float>(y[i]);
    }

    // Async transfers — DMA runs on stream while CPU continues
    CUDA_CHECK(cudaMemcpyAsync(d_xcoords, h_pinned_x,
                               n * sizeof(float),
                               cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_ycoords, h_pinned_y,
                               n * sizeof(float),
                               cudaMemcpyHostToDevice, stream));

    // Caller is responsible for synchronising the stream before launching
    // any kernel that reads d_xcoords / d_ycoords.

    nvtxRangePop();   
}

void instance_free_coords() noexcept
{
    nvtxRangePush("instance_free_coords");   

    if (d_xcoords) {
        CUDA_CHECK(cudaFree(d_xcoords));     
        d_xcoords = nullptr;
    }
    if (d_ycoords) {
        CUDA_CHECK(cudaFree(d_ycoords));     
        d_ycoords = nullptr;
    }
    if (h_pinned_x) {
        CUDA_CHECK(cudaFreeHost(h_pinned_x));
        h_pinned_x = nullptr;
    }
    if (h_pinned_y) {
        CUDA_CHECK(cudaFreeHost(h_pinned_y));
        h_pinned_y = nullptr;
    }

    d_n_vertices = 0;
    g_pinned_n   = 0;

    nvtxRangePop();   

void launch_batched_cost(
    const int2*  d_pairs,
    int*         d_out,
    int          n_pairs,
    cudaStream_t stream)
{
    if (n_pairs == 0) return;

    nvtxRangePushA("batched_cost_kernel");   

    // Query optimal block size once, cache in g_cost_block
    if (g_cost_block == 256) {   // sentinel: not yet queried
        int min_grid = 0;
        CUDA_CHECK(cudaOccupancyMaxPotentialBlockSize(
            &min_grid, &g_cost_block,
            batched_cost_kernel,
            0,      // dynamic shared memory bytes
            0));    // no block-size limit
        // g_cost_block is now the SM-optimal block size (typically 128 or 256)
    }

    const int grid = (n_pairs + g_cost_block - 1) / g_cost_block;

    batched_cost_kernel<<<grid, g_cost_block, 0, stream>>>(
        d_xcoords, d_ycoords, d_pairs, d_out, n_pairs);

    CUDA_CHECK(cudaGetLastError());

    nvtxRangePop();  
}

bool instance_gpu_ready() noexcept
{
    return (d_xcoords != nullptr) && (d_ycoords != nullptr) &&
           (d_n_vertices > 0);
}

} // namespace cobra

namespace cobra {

void Instance::gpu_upload() const
{
    // Use the default stream (0) for simplicity; callers that need async
    // behaviour should call instance_upload_coords() directly with a stream.
    instance_upload_coords(*this, /*stream=*/0);
    // Synchronise to ensure transfers complete before any kernel launches
    CUDA_CHECK(cudaStreamSynchronize(0));
}

void Instance::gpu_free() const noexcept
{
    instance_free_coords();
}

bool Instance::gpu_ready() const noexcept
{
    return instance_gpu_ready();
}

} 
