// opt/recreate_gpu.cu

#include "../base/gpu_utils.cuh"
#include <cuda_runtime.h>

extern float* d_xcoords;
extern float* d_ycoords;

// Candidate descriptor (aligned for coalescing)
struct alignas(16) InsertCandidate {
    int customer;
    int prev;
    int next;
    int pad;
};

// Device distance (squared distance)
__device__ __forceinline__
float dist2(const float* __restrict__ xc,
            const float* __restrict__ yc,
            int a, int b)
{
    float dx = xc[a] - xc[b];
    float dy = yc[a] - yc[b];
    return dx * dx + dy * dy;
}

// Kernel
static constexpr int REC_BLOCK = 256;

__global__ void recreate_kernel(
    const float* __restrict__ xc,
    const float* __restrict__ yc,
    const InsertCandidate* __restrict__ cands,
    float* __restrict__ out_delta,
    int n_cands)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_cands) return;

    const InsertCandidate c = cands[tid];

    // Old arc (prev → next)
    float old_arc = dist2(xc, yc, c.prev, c.next);

    // New arcs (prev → customer → next)
    float new_arcs =
        dist2(xc, yc, c.prev, c.customer) +
        dist2(xc, yc, c.customer, c.next);

    out_delta[tid] = new_arcs - old_arc;
}

void launch_recreate(
    const InsertCandidate* h_cands,
    float* h_deltas,
    int n_cands,
    cudaStream_t stream)
{
    if (n_cands == 0) return;

    size_t bytes_c = (size_t)n_cands * sizeof(InsertCandidate);
    size_t bytes_d = (size_t)n_cands * sizeof(float);

    InsertCandidate* d_cands = nullptr;
    float* d_delta = nullptr;

    CUDA_CHECK(cudaMalloc(&d_cands, bytes_c));
    CUDA_CHECK(cudaMalloc(&d_delta, bytes_d));

    CUDA_CHECK(cudaMemcpyAsync(
        d_cands, h_cands, bytes_c,
        cudaMemcpyHostToDevice, stream));

    int grid = (n_cands + REC_BLOCK - 1) / REC_BLOCK;

    recreate_kernel<<<grid, REC_BLOCK, 0, stream>>>(
        d_xcoords, d_ycoords,
        d_cands, d_delta, n_cands);

    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaMemcpyAsync(
        h_deltas, d_delta, bytes_d,
        cudaMemcpyDeviceToHost, stream));

    CUDA_CHECK(cudaStreamSynchronize(stream));

    CUDA_CHECK(cudaFree(d_cands));
    CUDA_CHECK(cudaFree(d_delta));
}