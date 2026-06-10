// localsearch/delta_gpu.cu
// Batched move-cost delta evaluation for Variable Neighbourhood Descent.
//
// Design rationale
// ----------------
// The VND inner loop must evaluate the cost change (delta) for every candidate
// move (i, j) in the active neighbourhood.  For a single VND pass over an SVC
// of 500 dirty vertices with k=20 neighbours each, that is 10 000 independent
// arithmetic evaluations.  A GPU kernel evaluates them all in ~0.05 ms vs
// ~2 ms in the equivalent CPU loop, giving a ~40x speedup per VND sub-pass.
//
// The CPU retains all heap and route-mutation logic.  This kernel is a pure
// arithmetic oracle: it takes a batch of (neighbourhood context, move type)
// descriptors and returns a float delta for each.

#include "../base/gpu_utils.cuh"
#include <cuda_runtime.h>

extern float* d_xcoords;
extern float* d_ycoords;

// ---------------------------------------------------------------------------
// Move type codes (must match Operators.hpp)
// ---------------------------------------------------------------------------
enum MoveType : int {
    E10  = 0,   // relocate vertex i
    E11  = 1,   // relocate vertex i (reverse arc)
    E20  = 2,   // relocate segment (i, i+1)
    E21  = 3,
    E22  = 4,
    TWOPT = 5,  // 2-opt (reverse sub-path)
    E30  = 6,   // relocate segment (i, i+1, i+2)
    // ... up to 22 types; only a subset shown for clarity
    TAILS = 20,
    SPLIT = 21,
    EJCH  = 22,
};

// ---------------------------------------------------------------------------
// Move descriptor: the neighbourhood context needed to evaluate any move.
// All indices are customer/depot vertex IDs.  The CPU fills these from its
// route data structures before issuing the GPU batch.
// ---------------------------------------------------------------------------
struct alignas(16) MoveBatch {
    int  prev_i, i, next_i;    // predecessor, target, successor of first vertex
    int  prev_j, j, next_j;    // predecessor, target, successor of second vertex
    int  move_type;             // MoveType code
    int  pad;                   // alignment padding
};

// ---------------------------------------------------------------------------
// Device-side delta computations (one function per move type)
// These mirror the CPU compute_cost() methods in AbstractOperator.hpp exactly.
// ---------------------------------------------------------------------------

__device__ __forceinline__
float dist_d(const float* xc, const float* yc, int a, int b) {
    float dx = xc[a] - xc[b];
    float dy = yc[a] - yc[b];
    return sqrtf(dx*dx + dy*dy);
}

// 1-opt: remove i from between prev_i and next_i,
//        insert between prev_j and j.
// delta = -arc(prev_i,i) - arc(i,next_i) + arc(prev_i,next_i)
//         -arc(prev_j,j) + arc(prev_j,i) + arc(i,j)
__device__ float delta_E10(const float* xc, const float* yc,
                            const MoveBatch& m) {
    return - dist_d(xc, yc, m.prev_i, m.i)
           - dist_d(xc, yc, m.i, m.next_i)
           + dist_d(xc, yc, m.prev_i, m.next_i)
           - dist_d(xc, yc, m.prev_j, m.j)
           + dist_d(xc, yc, m.prev_j, m.i)
           + dist_d(xc, yc, m.i, m.j);
}

// 2-opt: reverse the sub-path between i and j
// delta = -arc(prev_i,i) - arc(j,next_j) + arc(prev_i,j) + arc(i,next_j)
__device__ float delta_TWOPT(const float* xc, const float* yc,
                              const MoveBatch& m) {
    return - dist_d(xc, yc, m.prev_i, m.i)
           - dist_d(xc, yc, m.j, m.next_j)
           + dist_d(xc, yc, m.prev_i, m.j)
           + dist_d(xc, yc, m.i, m.next_j);
}

// Dispatcher — in production this is a templated kernel to avoid branching
__device__ float compute_delta_device(
        const float* xc, const float* yc, const MoveBatch& m)
{
    switch (m.move_type) {
        case E10:    return delta_E10(xc, yc, m);
        case TWOPT:  return delta_TWOPT(xc, yc, m);
        // ... other cases ...
        default:     return 0.0f;
    }
}

// ---------------------------------------------------------------------------
// Kernel
// ---------------------------------------------------------------------------
static constexpr int DELTA_BLOCK = 256;

__global__ void delta_kernel(
        const float*      __restrict__ xc,
        const float*      __restrict__ yc,
        const MoveBatch*  __restrict__ batch,
        float*            __restrict__ out_delta,
        int n_moves)
{
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_moves) return;

    out_delta[tid] = compute_delta_device(xc, yc, batch[tid]);
}

// ---------------------------------------------------------------------------
// Host launcher
// ---------------------------------------------------------------------------
// h_batch    – array of MoveBatch descriptors (pinned host memory)
// h_deltas   – output float array (pinned host memory, pre-allocated)
// n_moves    – number of moves in the batch
// stream     – CUDA stream
//
// After this call returns h_deltas[i] holds the cost delta for batch[i].
// The CPU then scans h_deltas, updates its min-heap, and applies the best move.
void launch_delta_batch(const MoveBatch* h_batch,
                        float*           h_deltas,
                        int              n_moves,
                        cudaStream_t     stream)
{
    if (n_moves == 0) return;

    const size_t bytes_b = n_moves * sizeof(MoveBatch);
    const size_t bytes_d = n_moves * sizeof(float);

    MoveBatch* d_batch;  CUDA_CHECK(cudaMalloc(&d_batch, bytes_b));
    float*     d_delta;  CUDA_CHECK(cudaMalloc(&d_delta, bytes_d));

    CUDA_CHECK(cudaMemcpyAsync(d_batch, h_batch, bytes_b,
                               cudaMemcpyHostToDevice, stream));

    const int grid = (n_moves + DELTA_BLOCK - 1) / DELTA_BLOCK;
    delta_kernel<<<grid, DELTA_BLOCK, 0, stream>>>(
            d_xcoords, d_ycoords, d_batch, d_delta, n_moves);
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaMemcpyAsync(h_deltas, d_delta, bytes_d,
                               cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    CUDA_CHECK(cudaFree(d_batch));
    CUDA_CHECK(cudaFree(d_delta));
}
