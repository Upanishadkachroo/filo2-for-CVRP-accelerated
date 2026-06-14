#ifndef CUDA_NEIGHBOR_FINDER_HPP
#define CUDA_NEIGHBOR_FINDER_HPP

#include <vector>

namespace cobra {

class CudaNeighborFinder {
public:
    CudaNeighborFinder(const std::vector<double>& x, const std::vector<double>& y);
    ~CudaNeighborFinder();

    // Compute k nearest neighbors for all vertices (GPU batched brute‑force)
    // Returns a flat array of size n * k, row-major (neighbors of vertex i start at i*k)
    std::vector<int> computeAllNeighborsFlat(int k, bool verbose = false);

    // Convenience wrapper: returns vector of vectors (only for small n, or for testing)
    std::vector<std::vector<int>> computeAllNeighbors(int k, bool verbose = false);

private:
    double* d_x = nullptr;
    double* d_y = nullptr;
    int n;
    std::vector<double> h_x, h_y;   // host copies
};

} // namespace cobra

#endif