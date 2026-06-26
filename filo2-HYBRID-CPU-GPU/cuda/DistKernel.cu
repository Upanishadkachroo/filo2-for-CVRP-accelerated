#include "DistKernel.cuh"
#include <cmath>

__global__ void distanceKernel(const float* __restrict__ xcoords,
                               const float* __restrict__ ycoords,
                               const int* __restrict__ neighbors,
                               const int* __restrict__ affected,
                               int n_affected,
                               int max_neighbors,
                               float* __restrict__ h_dist_cache)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = n_affected * max_neighbors;
    if (idx >= total) return;

    int a_idx = idx / max_neighbors;
    int slot   = idx % max_neighbors;

    int v = affected[a_idx];
    int j = neighbors[v * max_neighbors + slot];
    if (j < 0) {
        h_dist_cache[idx] = -1.0f;
        return;
    }

    float dx = xcoords[v] - xcoords[j];
    float dy = ycoords[v] - ycoords[j];
    float dist = roundf(sqrtf(dx*dx + dy*dy));
    h_dist_cache[idx] = dist;
}

void launchDistanceKernel(const float* xcoords_d,
                          const float* ycoords_d,
                          const int* neighbors_d,
                          const int* affected_d,
                          int n_affected,
                          int max_neighbors,
                          float* h_dist_cache,
                          cudaStream_t stream)
{
    int blockSize = 256;
    int gridSize = (n_affected * max_neighbors + blockSize - 1) / blockSize;
    distanceKernel<<<gridSize, blockSize, 0, stream>>>(
        xcoords_d, ycoords_d, neighbors_d,
        affected_d, n_affected, max_neighbors, h_dist_cache
    );
}
