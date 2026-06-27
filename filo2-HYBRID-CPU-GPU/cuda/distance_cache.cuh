#ifndef DISTANCE_CACHE_CUH
#define DISTANCE_CACHE_CUH

#include <cuda_runtime.h>

namespace cobra {

/**
 * @brief CUDA kernel that computes Euclidean distances from each affected vertex
 *        to all its neighbours (up to k+1 neighbours, including self).
 *
 * The output is written to pinned (page‑locked) host memory so the CPU can read
 * it directly after the kernel finishes.
 *
 * @param x           Device pointer to X coordinates (size N)
 * @param y           Device pointer to Y coordinates (size N)
 * @param neighbors   Flattened neighbour lists, size N * (k+1).
 *                    neighbours[v * (k+1) + slot] gives the vertex index.
 * @param k           Number of neighbours per vertex (excluding self) – the
 *                    actual list length is k+1 (self at slot 0).
 * @param affected    Device pointer to list of affected vertex indices (size num_affected)
 * @param num_affected Number of affected vertices
 * @param output      Pinned host memory, size num_affected * (k+1) floats.
 *                    output[vertex_index * (k+1) + slot] = distance from affected vertex
 *                    to neighbours[vertex_index][slot].
 */
__global__ void computeDistancesKernel(
    const float* __restrict__ x,
    const float* __restrict__ y,
    const int*   __restrict__ neighbors,
    int           k,
    const int*    __restrict__ affected,
    int           num_affected,
    float*        __restrict__ output
);

/**
 * @brief Helper to launch the kernel with optimal grid/block sizes.
 *
 * This function is called from the host `DistanceCache::compute()`.
 *
 * @param stream  CUDA stream on which to launch the kernel (async).
 */
void launchDistanceKernel(
    const float* x,
    const float* y,
    const int*   neighbors,
    int           k,
    const int*    affected,
    int           num_affected,
    float*        output,
    cudaStream_t  stream
);

} // namespace cobra

#endif // DISTANCE_CACHE_CUH
