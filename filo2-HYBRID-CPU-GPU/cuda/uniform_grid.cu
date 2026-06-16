#include "uniform_grid.cuh"

#include <cuda_runtime.h>
#include <thrust/sort.h>
#include <thrust/sequence.h>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <cfloat>
#include <cmath>
#include <algorithm>
#include <vector>
#include <cassert>

// -------------------------------------------------------------------
// Device helper: swap two values (for heap operations)
// -------------------------------------------------------------------
template <typename T>
__device__ void deviceSwap(T &a, T &b) {
    T tmp = a;
    a = b;
    b = tmp;
}

// -------------------------------------------------------------------
// Kernel: compute cell id for each point
// -------------------------------------------------------------------
__global__ void computeCellIdsKernel(const float* x, const float* y, int N,
                                     float minX, float minY, float cellSize,
                                     int gridY, int* cellIds) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;
    int cx = (int)floorf((x[idx] - minX) / cellSize);
    int cy = (int)floorf((y[idx] - minY) / cellSize);
    // Clamp (should be within bounds, but safeguard)
    cellIds[idx] = cx * gridY + cy;
}

// -------------------------------------------------------------------
// Device: push onto a max-heap (size ≤ k)
// -------------------------------------------------------------------
__device__ void heapPush(float* heapDist, int* heapIdx, int& heapSize, int k,
                         float dist, int idx) {
    if (heapSize < k) {
        heapSize++;
        int i = heapSize;
        heapDist[i] = dist;
        heapIdx[i] = idx;
        while (i > 1 && heapDist[i] > heapDist[i/2]) {
            deviceSwap(heapDist[i], heapDist[i/2]);
            deviceSwap(heapIdx[i], heapIdx[i/2]);
            i /= 2;
        }
    } else if (dist < heapDist[1]) {
        heapDist[1] = dist;
        heapIdx[1] = idx;
        int i = 1;
        while (i*2 <= heapSize) {
            int child = i*2;
            if (child+1 <= heapSize && heapDist[child+1] > heapDist[child]) child++;
            if (heapDist[child] > heapDist[i]) {
                deviceSwap(heapDist[i], heapDist[child]);
                deviceSwap(heapIdx[i], heapIdx[child]);
                i = child;
            } else break;
        }
    }
}

// -------------------------------------------------------------------
// Query kernel (exact kNN with ring expansion)
// -------------------------------------------------------------------
__global__ void queryKernel(int startIdx, int chunkSize, int k,
                            const float* x, const float* y,
                            const int* sortedIdx,
                            const int* cellStart, const int* cellCount,
                            int gridX, int gridY, float minX, float minY,
                            float cellSize,
                            float* heapDist, int* heapIdx,
                            int* outNeighbors) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= chunkSize) return;

    int qid = startIdx + tid;

    float* hDist = heapDist + tid * (k + 1);
    int*   hIdx  = heapIdx  + tid * (k + 1);
    int heapSize = 0;

    float qx = x[qid], qy = y[qid];

    int cx = (int)floorf((qx - minX) / cellSize);
    int cy = (int)floorf((qy - minY) / cellSize);
    cx = max(0, min(cx, gridX-1));
    cy = max(0, min(cy, gridY-1));

    int maxRing = max(gridX, gridY);
    for (int ring = 0; ring <= maxRing; ++ring) {
        int startX = max(cx - ring, 0), endX = min(cx + ring, gridX-1);
        int startY = max(cy - ring, 0), endY = min(cy + ring, gridY-1);

        for (int xCell = startX; xCell <= endX; ++xCell) {
            for (int yCell = startY; yCell <= endY; ++yCell) {
                int dx = xCell - cx, dy = yCell - cy;
                if (max(abs(dx), abs(dy)) != ring) continue;  // only ring boundary

                int cellId = xCell * gridY + yCell;
                int start = cellStart[cellId];
                int count = cellCount[cellId];
                for (int i = 0; i < count; ++i) {
                    int pIdx = sortedIdx[start + i];
                    float px = x[pIdx], py = y[pIdx];
                    float dx2 = qx - px, dy2 = qy - py;
                    float dist = dx2*dx2 + dy2*dy2;
                    heapPush(hDist, hIdx, heapSize, k, dist, pIdx);
                }
            }
        }

        // Termination check: if heap is full and distance to next ring >= kth distance
        if (heapSize == k) {
            float kthDist = hDist[1];
            // Distance from query point to the outer boundary of current square
            float left   = (cx - ring) * cellSize + minX;
            float right  = (cx + ring + 1) * cellSize + minX;
            float bottom = (cy - ring) * cellSize + minY;
            float top    = (cy + ring + 1) * cellSize + minY;
            float dLeft  = qx - left;
            float dRight = right - qx;
            float dBot   = qy - bottom;
            float dTop   = top - qy;
            float minDistToBoundary = min(min(dLeft, dRight), min(dBot, dTop));
            if (minDistToBoundary * minDistToBoundary >= kthDist) {
                break;
            }
        }
    }

    // Pop heap to output in ascending order
    int actualSize = heapSize;
    for (int i = 0; i < actualSize; ++i) {
        int outIdx = actualSize - 1 - i;
        outNeighbors[tid * k + outIdx] = hIdx[1];
        // Replace root with last
        hIdx[1] = hIdx[actualSize - i];
        hDist[1] = hDist[actualSize - i];
        int pos = 1;
        int currentSize = actualSize - i - 1;
        while (pos * 2 <= currentSize) {
            int child = pos * 2;
            if (child + 1 <= currentSize && hDist[child + 1] > hDist[child]) child++;
            if (hDist[child] > hDist[pos]) {
                deviceSwap(hDist[pos], hDist[child]);
                deviceSwap(hIdx[pos], hIdx[child]);
                pos = child;
            } else break;
        }
    }
    for (int i = actualSize; i < k; ++i) {
        outNeighbors[tid * k + i] = -1;
    }
}

// -------------------------------------------------------------------
// Host implementation
// -------------------------------------------------------------------
UniformGridNeighbors::UniformGridNeighbors()
    : N(0), d_x(nullptr), d_y(nullptr), d_sortedIdx(nullptr),
      d_cellStart(nullptr), d_cellCount(nullptr), gridBuilt(false) {}

UniformGridNeighbors::~UniformGridNeighbors() {
    freeDeviceData();
}

void UniformGridNeighbors::freeDeviceData() {
    cudaFree(d_x);
    cudaFree(d_y);
    cudaFree(d_sortedIdx);
    cudaFree(d_cellStart);
    cudaFree(d_cellCount);
    d_x = d_y = nullptr;
    d_sortedIdx = nullptr;
    d_cellStart = d_cellCount = nullptr;
    gridBuilt = false;
}

bool UniformGridNeighbors::allocateDeviceData(const std::vector<double>& x,
                                              const std::vector<double>& y) {
    N = x.size();
    if (N == 0) return false;

    cudaError_t err;
    err = cudaMalloc(&d_x, N * sizeof(float));
    if (err != cudaSuccess) return false;
    err = cudaMalloc(&d_y, N * sizeof(float));
    if (err != cudaSuccess) return false;

    std::vector<float> hx(N), hy(N);
    for (int i = 0; i < N; ++i) { hx[i] = (float)x[i]; hy[i] = (float)y[i]; }
    err = cudaMemcpy(d_x, hx.data(), N * sizeof(float), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) return false;
    err = cudaMemcpy(d_y, hy.data(), N * sizeof(float), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) return false;

    return true;
}

bool UniformGridNeighbors::build(const std::vector<double>& x,
                                 const std::vector<double>& y) {
    if (x.size() != y.size()) return false;
    N = x.size();
    if (N == 0) return false;

    // Bounding box
    minX = *std::min_element(x.begin(), x.end());
    maxX = *std::max_element(x.begin(), x.end());
    minY = *std::min_element(y.begin(), y.end());
    maxY = *std::max_element(y.begin(), y.end());

    float area = (maxX - minX) * (maxY - minY);
    float avgPointsPerCell = 30.0f;
    cellSize = sqrtf(area / (float)(N / avgPointsPerCell));
    cellSize = max(1e-6f, cellSize);

    gridX = (int)ceilf((maxX - minX) / cellSize);
    gridY = (int)ceilf((maxY - minY) / cellSize);
    gridX = max(1, gridX);
    gridY = max(1, gridY);
    numCells = gridX * gridY;

    // Allocate device memory for coordinates
    if (!allocateDeviceData(x, y)) {
        freeDeviceData();
        return false;
    }

    // Compute cell IDs
    int* d_cellIds = nullptr;
    cudaMalloc(&d_cellIds, N * sizeof(int));
    int blockSize = 256;
    int gridSize = (N + blockSize - 1) / blockSize;
    computeCellIdsKernel<<<gridSize, blockSize>>>(d_x, d_y, N, minX, minY,
                                                  cellSize, gridY, d_cellIds);
    cudaDeviceSynchronize();

    // Allocate sorted indices
    cudaMalloc(&d_sortedIdx, N * sizeof(int));

    // Use thrust to sort by cell ID
    auto cellIdsPtr = thrust::device_pointer_cast(d_cellIds);
    auto sortedIdxPtr = thrust::device_pointer_cast(d_sortedIdx);
    thrust::sequence(thrust::device, sortedIdxPtr, sortedIdxPtr + N);
    thrust::sort_by_key(thrust::device, cellIdsPtr, cellIdsPtr + N, sortedIdxPtr);

    // Build CSR on the host (simple and reliable)
    std::vector<int> h_cellIds(N);
    cudaMemcpy(h_cellIds.data(), d_cellIds, N * sizeof(int), cudaMemcpyDeviceToHost);

    std::vector<int> h_countAll(numCells, 0);
    for (int i = 0; i < N; ++i) {
        int cid = h_cellIds[i];
        if (cid >= 0 && cid < numCells) {
            h_countAll[cid]++;
        } else {
            // Clamp for safety (should not happen)
            cid = (cid < 0) ? 0 : numCells - 1;
            h_countAll[cid]++;
        }
    }

    // Prefix sum to get start offsets
    std::vector<int> h_start(numCells);
    int running = 0;
    for (int i = 0; i < numCells; ++i) {
        h_start[i] = running;
        running += h_countAll[i];
    }

    // Allocate and copy to device
    cudaMalloc(&d_cellCount, numCells * sizeof(int));
    cudaMalloc(&d_cellStart, numCells * sizeof(int));
    cudaMemcpy(d_cellCount, h_countAll.data(), numCells * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_cellStart, h_start.data(), numCells * sizeof(int), cudaMemcpyHostToDevice);

    cudaFree(d_cellIds);
    gridBuilt = true;
    return true;
}

bool UniformGridNeighbors::computeNeighbors(int k, std::vector<int>& outNeighbors) {
    if (!gridBuilt || N == 0) return false;
    outNeighbors.resize(N * k);

    const int CHUNK_SIZE = 2048;
    float* d_heapDist = nullptr;
    int* d_heapIdx = nullptr;
    int* d_out = nullptr;

    cudaMalloc(&d_heapDist, CHUNK_SIZE * (k + 1) * sizeof(float));
    cudaMalloc(&d_heapIdx, CHUNK_SIZE * (k + 1) * sizeof(int));
    cudaMalloc(&d_out, CHUNK_SIZE * k * sizeof(int));

    for (int start = 0; start < N; start += CHUNK_SIZE) {
        int chunkSize = std::min(CHUNK_SIZE, N - start);
        int blockSize = 256;
        int gridSize = (chunkSize + blockSize - 1) / blockSize;
        queryKernel<<<gridSize, blockSize>>>(start, chunkSize, k,
                                             d_x, d_y,
                                             d_sortedIdx,
                                             d_cellStart, d_cellCount,
                                             gridX, gridY, minX, minY, cellSize,
                                             d_heapDist, d_heapIdx,
                                             d_out);
        cudaDeviceSynchronize();
        std::vector<int> h_out(chunkSize * k);
        cudaMemcpy(h_out.data(), d_out, chunkSize * k * sizeof(int), cudaMemcpyDeviceToHost);
        for (int i = 0; i < chunkSize; ++i) {
            int base = (start + i) * k;
            for (int j = 0; j < k; ++j) {
                outNeighbors[base + j] = h_out[i * k + j];
            }
        }
    }

    cudaFree(d_heapDist);
    cudaFree(d_heapIdx);
    cudaFree(d_out);
    return true;
}
