// cuda/distance_cache.cu
#include "distance_cache.cuh"
#include <cuda_runtime.h>

namespace cobra {

DistanceCache::DistanceCache(const Instance& inst, int k_)
    : instance(inst), k(k_), max_vertices(k_+1) {
    cudaStreamCreate(&stream);
    // Allocate pinned memory for output: large enough for the max expected affected vertices.
    // We can over‑allocate (e.g., 256 * (k+1)) to avoid repeated allocations.
    size_t max_affected = 256;  // more than enough
    cudaHostAlloc(&d_output, max_affected * (k+1) * sizeof(float), cudaHostAllocDefault);
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
    // Copy affected vertices to device
    cudaMemcpyAsync(d_affected, vertices.data(), n * sizeof(int), cudaMemcpyHostToDevice, stream);

    // Launch kernel
    int total_threads = n * (k+1);
    int blockSize = 256;
    int gridSize = (total_threads + blockSize - 1) / blockSize;
    computeDistancesKernel<<<gridSize, blockSize, 0, stream>>>(
        instance.get_xcoords_gpu(),  // assuming we have GPU copies of coords
        instance.get_ycoords_gpu(),
        instance.get_neighbors_gpu(), // flattened neighbor lists
        k,
        d_affected,
        n,
        d_output
    );
    // Note: kernel is async; we will synchronize later.
}

void DistanceCache::synchronize() {
    cudaStreamSynchronize(stream);

    // After sync, build the host map from the pinned memory
    cache_map.clear();
    int n = current_vertices.size();
    const float* data = d_output;
    for (int idx = 0; idx < n; ++idx) {
        int v = current_vertices[idx];
        auto& map = cache_map[v];
        const auto& nbrs = instance.get_neighbors_of(v);
        for (int slot = 0; slot < (int)nbrs.size(); ++slot) {
            int j = nbrs[slot];
            float dist = data[idx * (k+1) + slot];
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
    return -1.0f; // not cached
}

} // namespace cobra
