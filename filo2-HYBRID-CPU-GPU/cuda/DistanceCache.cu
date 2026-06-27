#include "DistanceCache.hpp"
#include "distance_cache.cuh"
#include "../instance/Instance.hpp"
#include <cuda_runtime.h>
#include <cmath>       // for sqrtf
#include <cstring>

namespace cobra {

// ─── Kernel implementation ──────────────────────────────────────────────────
__global__ void computeDistancesKernel(
    const float* __restrict__ x,
    const float* __restrict__ y,
    const int*   __restrict__ neighbors,
    int           k,
    const int*    __restrict__ affected,
    int           num_affected,
    float*        __restrict__ output
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total_threads = num_affected * (k + 1);
    if (tid >= total_threads) return;

    int vertex_idx = tid / (k + 1);
    int slot = tid % (k + 1);
    int v = affected[vertex_idx];
    int j = neighbors[v * (k + 1) + slot];

    float dx = x[v] - x[j];
    float dy = y[v] - y[j];
    output[tid] = sqrtf(dx * dx + dy * dy);
}

// ─── DistanceCache host methods ────────────────────────────────────────────
DistanceCache::DistanceCache(const Instance& inst, int k_)
    : m_instance(&inst),
      xcoords(inst.get_xcoords()),
      ycoords(inst.get_ycoords()),
      k(k_),
      max_affected(256) {
    cudaStreamCreate(&stream);
    cudaHostAlloc(&d_output, max_affected * (k + 1) * sizeof(float), cudaHostAllocDefault);
    cudaMalloc(&d_affected, max_affected * sizeof(int));
}

DistanceCache::~DistanceCache() {
    cudaFree(d_affected);
    cudaFreeHost(d_output);
    cudaStreamDestroy(stream);
}

void DistanceCache::compute(const std::vector<int>& vertices) {
    if (vertices.empty()) return;
    current_vertices = vertices;
    int n = vertices.size();
    if (n > max_affected) {
        // Reallocate
        cudaFree(d_affected);
        cudaFreeHost(d_output);
        max_affected = n * 2;
        cudaHostAlloc(&d_output, max_affected * (k + 1) * sizeof(float), cudaHostAllocDefault);
        cudaMalloc(&d_affected, max_affected * sizeof(int));
    }
    cudaMemcpyAsync(d_affected, vertices.data(), n * sizeof(int), cudaMemcpyHostToDevice, stream);

    const float* d_x = m_instance->get_xcoords_gpu();
    const float* d_y = m_instance->get_ycoords_gpu();
    const int* d_neighbors = m_instance->get_neighbors_flat_gpu();

    int total_threads = n * (k + 1);
    int blockSize = 256;
    int gridSize = (total_threads + blockSize - 1) / blockSize;
    computeDistancesKernel<<<gridSize, blockSize, 0, stream>>>(
        d_x, d_y, d_neighbors, k,
        d_affected, n,
        d_output
    );
}

void DistanceCache::synchronize() {
    cudaStreamSynchronize(stream);

    cache_map.clear();
    int n = current_vertices.size();
    const float* data = d_output;
    for (int idx = 0; idx < n; ++idx) {
        int v = current_vertices[idx];
        auto& map = cache_map[v];
        const auto& nbrs = m_instance->get_neighbors_of(v);
        for (int slot = 0; slot < static_cast<int>(nbrs.size()); ++slot) {
            int j = nbrs[slot];
            float dist = data[idx * (k + 1) + slot];
            map[j] = dist;
        }
    }
}

float DistanceCache::get_distance(int i, int j) const {
    auto it_v = cache_map.find(i);
    if (it_v != cache_map.end()) {
        auto it_n = it_v->second.find(j);
        if (it_n != it_v->second.end())
            return it_n->second;
    }
    return -1.0f;
}

} // namespace cobra
