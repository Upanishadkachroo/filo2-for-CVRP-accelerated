#ifndef DISTANCE_CACHE_HPP
#define DISTANCE_CACHE_HPP

#include <vector>
#include <unordered_map>
#include <cuda_runtime.h>

namespace cobra {
    class Instance;
}

namespace cobra {

class DistanceCache {
public:
    DistanceCache(const Instance& inst, int k);
    ~DistanceCache();

    void compute(const std::vector<int>& vertices);
    void synchronize();
    float get_distance(int i, int j) const;

private:
    const Instance* m_instance;          // pointer to instance for GPU data & neighbor lists
    const std::vector<double>& xcoords;   // host coordinates (not used except for reference)
    const std::vector<double>& ycoords;
    int k;
    int max_affected;

    float* d_output;      // pinned host memory
    int*   d_affected;    // device copy of affected list
    cudaStream_t stream;

    std::unordered_map<int, std::unordered_map<int, float>> cache_map;
    std::vector<int> current_vertices;
};

} // namespace cobra

#endif
