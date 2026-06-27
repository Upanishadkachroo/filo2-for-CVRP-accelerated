#ifndef DISTANCE_CACHE_CUH
#define DISTANCE_CACHE_CUH

#include <cuda_runtime.h>

namespace cobra {

__global__ void computeDistancesKernel(
    const float* __restrict__ x,
    const float* __restrict__ y,
    const int*   __restrict__ neighbors,
    int           k,
    const int*    __restrict__ affected,
    int           num_affected,
    float*        __restrict__ output
);

} // namespace cobra

#endif
