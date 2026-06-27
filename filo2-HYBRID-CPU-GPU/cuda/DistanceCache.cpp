#include "DistanceCache.hpp"
#include "../instance/Instance.hpp" 
#include "../instance/Instance.hpp"
#include "distance_cache.cuh"   // kernel declaration
#include <cuda_runtime.h>
#include <cstring>
#include <iostream>

namespace cobra {

DistanceCache::DistanceCache(const Instance& inst, int k_)
    : xcoords(inst.get_xcoords()), ycoords(inst.get_ycoords()),
      neighbors(inst.get_neighbors_of()), k(k_), max_affected(256) {
    cudaStreamCreate(&stream);
    // Allocate pinned memory for output: enough for max_affected vertices.
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
    // If n exceeds max_affected, we could reallocate, but for simplicity we assume it fits.
    if (n > max_affected) {
        // (optional) reallocate – not expected in practice.
    }
    // Copy affected vertices to device
    cudaMemcpyAsync(d_affected, vertices.data(), n * sizeof(int), cudaMemcpyHostToDevice, stream);

    // Flatten neighbor lists: we need a flat array of size N * (k+1).
    // This assumes the instance's neighbor lists are stored as vector<vector<int>>.
    // For performance, we should have a flat GPU copy already. For now, we'll pass
    // the flat pointer from the instance (assuming we have one).
    // We'll need a separate method to get the flat device pointer.
    // This is a placeholder – we assume Instance provides a method get_neighbors_flat_gpu().
    // We'll implement that later.
    // For now, we'll use a temporary: we'll pass the raw data from the host (slow).
    // We'll improve later.

    // Launch kernel
    int total_threads = n * (k+1);
    int blockSize = 256;
    int gridSize = (total_threads + blockSize - 1) / blockSize;

    // We need device pointers to xcoords, ycoords, and flat neighbors.
    // We assume these are already on the device and accessible via Instance methods.
    // We'll add getters in Instance later.
    const float* d_x = instance.get_xcoords_gpu();  // need to add these to Instance
    const float* d_y = instance.get_ycoords_gpu();
    const int* d_neighbors = instance.get_neighbors_flat_gpu();

    computeDistancesKernel<<<gridSize, blockSize, 0, stream>>>(
        d_x, d_y, d_neighbors, k,
        d_affected, n,
        d_output
    );
}

void DistanceCache::synchronize() {
    cudaStreamSynchronize(stream);

    // Build host map from the pinned output
    cache_map.clear();
    int n = current_vertices.size();
    const float* data = d_output;
    for (int idx = 0; idx < n; ++idx) {
        int v = current_vertices[idx];
        auto& map = cache_map[v];
        const auto& nbrs = neighbors[v];  // from instance
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
    return -1.0f;
}

} // namespace cobra
