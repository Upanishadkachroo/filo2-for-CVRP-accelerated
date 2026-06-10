// localsearch/feasibility_gpu.cu
// Batched vehicle-capacity feasibility checking for proposed route merges.
//
// Design rationale
// ----------------
// Several operators (SPLIT, TAILS, EJCH) generate many candidate route merges
// in a single iteration. Each merge is feasible iff the combined load of
// customers assigned to the new route does not exceed vehicle capacity.
//
// Evaluating this as a CPU inner loop serialises thousands of comparisons.
// This kernel parallelises the checks.  The kernel is deliberately trivial —
// its value is removing the loop from the CPU hot path, not in complex
// arithmetic.

#include "../base/gpu_utils.cuh"
#include <cuda_runtime.h>

static constexpr int FEAS_BLOCK = 512;

// ---------------------------------------------------------------------------
// Kernel
// ---------------------------------------------------------------------------

/// For each candidate route merge, check whether the proposed combined load
/// fits within the vehicle capacity.
///
/// cand_loads[t]  – proposed total demand for candidate t
/// capacity       – vehicle capacity (same for all routes)
/// out_feasible[t]– true if cand_loads[t] <= capacity
__global__ void feasibility_kernel(
        const int*   __restrict__ cand_loads,
        int          capacity,
        bool*        __restrict__ out_feasible,
        int          n_cands)
{
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_cands) return;
    out_feasible[tid] = (cand_loads[tid] <= capacity);
}

// ---------------------------------------------------------------------------
// Host launcher
// ---------------------------------------------------------------------------

/// h_cand_loads  – proposed loads for each candidate merge (host array)
/// capacity      – vehicle capacity
/// h_feasible    – output bool array (pre-allocated, length n_cands)
/// n_cands       – number of candidates
/// stream        – CUDA stream
void launch_feasibility(const int* h_cand_loads,
                        int        capacity,
                        bool*      h_feasible,
                        int        n_cands,
                        cudaStream_t stream)
{
    if (n_cands == 0) return;

    int*  d_loads;    CUDA_CHECK(cudaMalloc(&d_loads,    n_cands * sizeof(int)));
    bool* d_feasible; CUDA_CHECK(cudaMalloc(&d_feasible, n_cands * sizeof(bool)));

    CUDA_CHECK(cudaMemcpyAsync(d_loads, h_cand_loads, n_cands * sizeof(int),
                               cudaMemcpyHostToDevice, stream));

    const int grid = (n_cands + FEAS_BLOCK - 1) / FEAS_BLOCK;
    feasibility_kernel<<<grid, FEAS_BLOCK, 0, stream>>>(
            d_loads, capacity, d_feasible, n_cands);
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaMemcpyAsync(h_feasible, d_feasible, n_cands * sizeof(bool),
                               cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    CUDA_CHECK(cudaFree(d_loads));
    CUDA_CHECK(cudaFree(d_feasible));
}
