#ifndef FILO2_DIST_KERNEL_CUH
#define FILO2_DIST_KERNEL_CUH

#include <cuda_runtime.h>

// Launch the distance precomputation kernel on GPU.
// xcoords_d, ycoords_d, neighbors_d are device pointers to coordinates and flat neighbor lists.
// affected_d is device pointer to list of affected vertex indices (n_affected).
// h_dist_cache is pinned host memory of size n_affected * max_neighbors.
// The kernel executes asynchronously on the given stream.
void launchDistanceKernel(const float* xcoords_d,
                          const float* ycoords_d,
                          const int* neighbors_d,
                          const int* affected_d,
                          int n_affected,
                          int max_neighbors,
                          float* h_dist_cache,
                          cudaStream_t stream);

#endif
