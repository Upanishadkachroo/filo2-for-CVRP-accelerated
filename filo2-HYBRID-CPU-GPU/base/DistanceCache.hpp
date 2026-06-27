// base/DistanceCache.hpp
#ifndef DISTANCE_CACHE_HPP
#define DISTANCE_CACHE_HPP

#include <vector>
#include <unordered_map>
#include <cuda_runtime.h>
#include "Instance.hpp"

namespace cobra {

class DistanceCache {
public:
    DistanceCache(const Instance& inst, int k);
    ~DistanceCache();

    // Compute distances for a set of vertices (the SVC).
    // This launches the GPU kernel asynchronously on the internal stream.
    void compute(const std::vector<int>& vertices);

    // Synchronize with the GPU stream (ensures kernel has finished).
    void synchronize();

    // Look up a cached distance. Returns -1.0 if not found.
    float get_distance(int i, int j) const;

    // Access the underlying pinned memory (for debugging).
    const float* get_data() const { return d_output; }

private:
    const Instance& instance;
    int k;
    int max_vertices;       // size of output per vertex = k+1

    // GPU data
    float* d_output;        // pinned host memory, size max_vertices * (k+1)
    int*   d_affected;      // device copy of affected vertices
    cudaStream_t stream;

    // Host‑side mapping: for each affected vertex, a map from neighbor -> distance.
    // This is built after the kernel completes.
    std::unordered_map<int, std::unordered_map<int, float>> cache_map;
    std::vector<int> current_vertices;
};

} // namespace cobra
#endif
