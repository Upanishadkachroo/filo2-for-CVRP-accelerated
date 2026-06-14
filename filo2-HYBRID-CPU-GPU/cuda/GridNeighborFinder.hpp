#ifndef GRID_NEIGHBOR_FINDER_HPP
#define GRID_NEIGHBOR_FINDER_HPP

#include <vector>

namespace cobra {

class GridNeighborFinder {
public:
    GridNeighborFinder(const std::vector<double>& x, const std::vector<double>& y);
    ~GridNeighborFinder();

    std::vector<int> computeAllNeighborsFlat(int k, bool verbose = false);
    std::vector<std::vector<int>> computeAllNeighbors(int k, bool verbose = false);

private:
    float* d_x = nullptr;
    float* d_y = nullptr;
    int n;
    std::vector<double> h_x, h_y;
    float min_x, max_x, min_y, max_y;
    float cell_size;
    int grid_w, grid_h, grid_cells;

    int* d_cell_offsets = nullptr;
    int* d_cell_data = nullptr;
    int* d_cell_counts = nullptr;

    void buildGrid();
};

} // namespace cobra

#endif