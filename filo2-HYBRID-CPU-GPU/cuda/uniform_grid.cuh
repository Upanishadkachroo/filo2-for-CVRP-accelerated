#ifndef UNIFORM_GRID_CUH
#define UNIFORM_GRID_CUH

#include <vector>
#include <cuda_runtime.h>

class UniformGridNeighbors {
public:
    UniformGridNeighbors();
    ~UniformGridNeighbors();

    // Build grid from host vectors (double precision input)
    bool build(const std::vector<double>& x, const std::vector<double>& y);

    // Compute k nearest neighbors for all points.
    // Output: flat array of size N * k (row i starts at i*k).
    // Returns true on success.
    bool computeNeighbors(int k, std::vector<int>& outNeighbors);

private:
    int N;                // number of points
    float* d_x;           // device x coordinates (float)
    float* d_y;           // device y coordinates
    int* d_sortedIdx;     // sorted point indices by cell
    int* d_cellStart;     // start offset per cell
    int* d_cellCount;     // count per cell
    int gridX, gridY;     // grid dimensions
    int numCells;
    float minX, maxX, minY, maxY;
    float cellSize;

    // Device memory for the grid (persistent after build)
    bool gridBuilt;

    // Helper: allocate and copy host data
    bool allocateDeviceData(const std::vector<double>& x, const std::vector<double>& y);
    void freeDeviceData();
};

#endif