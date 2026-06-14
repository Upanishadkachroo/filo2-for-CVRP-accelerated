#ifndef CUDA_NEIGHBOR_FINDER_HPP
#define CUDA_NEIGHBOR_FINDER_HPP

#include <vector>

namespace cobra {

/**
 * Brute‑force CUDA k‑nearest neighbour finder.
 * Suitable for instances up to ~200,000 vertices.
 * For larger instances, use KDTree+OpenMP fallback.
 */
class CudaNeighborFinder {
public:
    /**
     * Constructor – copies coordinates to GPU.
     * @param x vector of x coordinates (double)
     * @param y vector of y coordinates (double)
     */
    CudaNeighborFinder(const std::vector<double>& x, const std::vector<double>& y);

    /// Destructor – frees GPU memory.
    ~CudaNeighborFinder();

    /**
     * Compute k nearest neighbours for all vertices.
     * @param k number of neighbours
     * @param verbose print timing
     * @return flat array of size n * k (row‑major: neighbours of vertex i start at i*k)
     */
    std::vector<int> computeAllNeighborsFlat(int k, bool verbose = false);

    /**
     * Compute k nearest neighbours for all vertices.
     * @param k number of neighbours
     * @param verbose print timing
     * @return vector of vectors (neighbours[i] = list of indices)
     */
    std::vector<std::vector<int>> computeAllNeighbors(int k, bool verbose = false);

private:
    float* d_x = nullptr;     // device x coordinates (float)
    float* d_y = nullptr;     // device y coordinates (float)
    int n;                    // number of vertices
    std::vector<double> h_x;  // host copy (needed for potential reuse)
    std::vector<double> h_y;
};

} // namespace cobra

#endif