#ifndef GRID_NEIGHBOR_FINDER_HPP
#define GRID_NEIGHBOR_FINDER_HPP

#include <vector>

namespace cobra {

/**
 * Grid‑based CUDA k‑nearest neighbour finder.
 * Builds a uniform grid (CSR) on the GPU, then for each query expands
 * cells in rings until at least k candidates are found, and finally
 * selects the k smallest distances.
 *
 * Suitable for large instances (up to 1M vertices) with typical
 * spatial distributions.
 */
class GridNeighborFinder {
public:
    GridNeighborFinder(const std::vector<double>& x, const std::vector<double>& y);
    ~GridNeighborFinder();

    // Returns a flat array of size n * k (row‑major)
    std::vector<int> computeAllNeighborsFlat(int k, bool verbose = false);

    // Returns vector of vectors (convenience wrapper)
    std::vector<std::vector<int>> computeAllNeighbors(int k, bool verbose = false);

private:
    float* d_x = nullptr;
    float* d_y = nullptr;
    int n;
    std::vector<double> h_x, h_y;
    float min_x, max_x, min_y, max_y;
    float cell_size;
    int grid_w, grid_h, grid_cells;

    int* d_cell_offsets = nullptr;   // prefix sum of cell counts (size grid_cells+1)
    int* d_cell_data = nullptr;      // concatenated vertex indices (size n)

    void buildGrid();
};

} // namespace cobra

#endif