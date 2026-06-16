#include "uniform_grid.cuh"
#include <thrust/sort.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cassert>

// ------------------------------------------------
// Helper kernels for building the grid
// ------------------------------------------------

__global__ void computeCellIds(const float* x, const float* y, int N,
                               float minX, float minY, float cellSize,
                               int gridY, int* cellIds) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;
    int cx = (int)floorf((x[idx] - minX) / cellSize);
    int cy = (int)floorf((y[idx] - minY) / cellSize);
    // clamp (should not happen if bounding box is correct)
    // we can clamp to [0, gridX-1] etc., but we know bounds.
    cellIds[idx] = cx * gridY + cy;
}

// ... (other kernels for building CSR, etc.)

// ------------------------------------------------
// Query kernel – exact kNN using heap
// ------------------------------------------------

__device__ void heapPush(float* heapDist, int* heapIdx, int& heapSize, int k,
                         float dist, int idx) {
    if (heapSize < k) {
        heapSize++;
        int i = heapSize;
        heapDist[i] = dist;
        heapIdx[i] = idx;
        while (i > 1 && heapDist[i] > heapDist[i/2]) {
            swap(heapDist[i], heapDist[i/2]);
            swap(heapIdx[i], heapIdx[i/2]);
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
                swap(heapDist[i], heapDist[child]);
                swap(heapIdx[i], heapIdx[child]);
                i = child;
            } else break;
        }
    }
}

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

    int qid = startIdx + tid;   // global point index

    // Each thread has its own heap segment
    float* hDist = heapDist + tid * (k + 1);
    int*   hIdx  = heapIdx  + tid * (k + 1);
    int heapSize = 0;

    // Query point
    float qx = x[qid], qy = y[qid];

    // Cell coordinates
    int cx = (int)floorf((qx - minX) / cellSize);
    int cy = (int)floorf((qy - minY) / cellSize);
    // Clamp to grid bounds (should be within)
    cx = max(0, min(cx, gridX-1));
    cy = max(0, min(cy, gridY-1));

    // Ring expansion
    int maxRing = max(gridX, gridY); // worst case
    for (int ring = 0; ring <= maxRing; ++ring) {
        // Iterate cells on the ring (Chebyshev distance == ring)
        int startX = cx - ring, endX = cx + ring;
        int startY = cy - ring, endY = cy + ring;
        for (int dx = -ring; dx <= ring; ++dx) {
            int xCell = cx + dx;
            if (xCell < 0 || xCell >= gridX) continue;
            // For each x, we need y such that max(|dx|,|dy|) == ring
            // That means |dy| == ring OR |dx| == ring
            // We can iterate dy = -ring..ring, but skip the ones where both |dx| < ring and |dy| < ring
            // Simpler: iterate all y in [-ring, ring] and check max(abs(dx), abs(dy)) == ring
            for (int dy = -ring; dy <= ring; ++dy) {
                if (max(abs(dx), abs(dy)) != ring) continue;
                int yCell = cy + dy;
                if (yCell < 0 || yCell >= gridY) continue;
                int cellId = xCell * gridY + yCell;
                int start = cellStart[cellId];
                int count = cellCount[cellId];
                for (int i = 0; i < count; ++i) {
                    int pIdx = sortedIdx[start + i];
                    float px = x[pIdx], py = y[pIdx];
                    float dx = qx - px, dy = qy - py;
                    float dist = dx*dx + dy*dy;
                    heapPush(hDist, hIdx, heapSize, k, dist, pIdx);
                }
            }
        }

        // Check termination condition if heap is full
        if (heapSize == k) {
            float kthDist = hDist[1]; // max distance in max-heap
            // Compute minimum distance to the next ring (ring+1)
            // The current square (Chebyshev radius ring) has bounds:
            float left   = (cx - ring) * cellSize + minX;
            float right  = (cx + ring + 1) * cellSize + minX; // exclusive
            float bottom = (cy - ring) * cellSize + minY;
            float top    = (cy + ring + 1) * cellSize + minY;
            float dLeft  = qx - left;
            float dRight = right - qx;
            float dBot   = qy - bottom;
            float dTop   = top - qy;
            float minDistToBoundary = min(min(dLeft, dRight), min(dBot, dTop));
            // minDistToBoundary is distance to the square boundary; the next ring is outside
            float nextRingMinDist = minDistToBoundary;
            if (nextRingMinDist * nextRingMinDist >= kthDist) {
                // No point in the next ring (or farther) can be closer
                break;
            }
        }
    }

    // Now we have the heap (max-heap) containing up to k nearest.
    // We need to output in ascending distance order.
    // We can pop elements from the heap (max) and store in reverse order.
    int actualSize = heapSize;
    // We'll store in a temporary array (local) of size k, then copy to output.
    // Since k is small, we can use registers or local memory.
    // For simplicity, we'll write directly to global output in reverse order.
    for (int i = 0; i < actualSize; ++i) {
        // Pop max (at index 1)
        int outIdx = actualSize - 1 - i; // we want ascending, so the last popped is the smallest
        outNeighbors[tid * k + outIdx] = hIdx[1];
        // Replace root with last element
        hIdx[1] = hIdx[actualSize - i];
        hDist[1] = hDist[actualSize - i];
        // Bubble down
        int pos = 1;
        int currentSize = actualSize - i - 1;
        while (pos * 2 <= currentSize) {
            int child = pos * 2;
            if (child + 1 <= currentSize && hDist[child + 1] > hDist[child]) child++;
            if (hDist[child] > hDist[pos]) {
                swap(hDist[pos], hDist[child]);
                swap(hIdx[pos], hIdx[child]);
                pos = child;
            } else break;
        }
    }
    // If actualSize < k, fill remaining with -1 (should not happen if N >= k)
    for (int i = actualSize; i < k; ++i) {
        outNeighbors[tid * k + i] = -1;
    }
}

// ------------------------------------------------
// Host implementation
// ------------------------------------------------

UniformGridNeighbors::UniformGridNeighbors() : N(0), d_x(nullptr), d_y(nullptr),
    d_sortedIdx(nullptr), d_cellStart(nullptr), d_cellCount(nullptr),
    gridBuilt(false) {}

UniformGridNeighbors::~UniformGridNeighbors() { freeDeviceData(); }

bool UniformGridNeighbors::build(const std::vector<double>& x, const std::vector<double>& y) {
    N = x.size();
    if (N == 0) return false;

    // Compute bounding box
    minX = *std::min_element(x.begin(), x.end());
    maxX = *std::max_element(x.begin(), x.end());
    minY = *std::min_element(y.begin(), y.end());
    maxY = *std::max_element(y.begin(), y.end());

    // Determine cell size: aim for ~30 points per cell on average
    float area = (maxX - minX) * (maxY - minY);
    float avgPointsPerCell = 30.0f;
    cellSize = sqrtf(area / (N / avgPointsPerCell));
    // Clamp to avoid extreme values
    cellSize = max(1e-6f, cellSize);

    gridX = (int)ceilf((maxX - minX) / cellSize);
    gridY = (int)ceilf((maxY - minY) / cellSize);
    // Ensure at least 1 cell
    gridX = max(1, gridX);
    gridY = max(1, gridY);
    numCells = gridX * gridY;

    // Allocate device memory for coordinates (float) and cell IDs
    cudaError_t err;
    err = cudaMalloc(&d_x, N * sizeof(float));
    if (err != cudaSuccess) return false;
    err = cudaMalloc(&d_y, N * sizeof(float));
    if (err != cudaSuccess) return false;
    // Copy from double to float via temporary host vector
    std::vector<float> hx(N), hy(N);
    for (int i = 0; i < N; ++i) { hx[i] = (float)x[i]; hy[i] = (float)y[i]; }
    cudaMemcpy(d_x, hx.data(), N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, hy.data(), N * sizeof(float), cudaMemcpyHostToDevice);

    // Compute cell IDs
    int* d_cellIds;
    cudaMalloc(&d_cellIds, N * sizeof(int));
    int blockSize = 256;
    int gridSize = (N + blockSize - 1) / blockSize;
    computeCellIds<<<gridSize, blockSize>>>(d_x, d_y, N, minX, minY, cellSize, gridY, d_cellIds);
    cudaDeviceSynchronize();

    // Sort points by cell ID using thrust
    thrust::device_ptr<int> d_cellIds_ptr(d_cellIds);
    thrust::device_ptr<int> d_sortedIdx_ptr; // we need to allocate sortedIdx
    cudaMalloc(&d_sortedIdx, N * sizeof(int));
    // Fill sortedIdx with 0..N-1
    thrust::sequence(thrust::device, d_sortedIdx_ptr, d_sortedIdx_ptr + N);
    // Sort by cell ID
    thrust::sort_by_key(thrust::device, d_cellIds_ptr, d_cellIds_ptr + N, d_sortedIdx_ptr);

    // Now build CSR: cellStart and cellCount
    cudaMalloc(&d_cellStart, numCells * sizeof(int));
    cudaMalloc(&d_cellCount, numCells * sizeof(int));
    // Initialize counts to zero
    cudaMemset(d_cellCount, 0, numCells * sizeof(int));
    // We can use a kernel to count occurrences, or use thrust::reduce_by_key.
    // For simplicity, we copy cellIds to host, but that's inefficient for large N.
    // Better: use thrust::reduce_by_key
    thrust::device_vector<int> cellIdsVec(d_cellIds, d_cellIds + N);
    thrust::device_vector<int> sortedIdxVec(d_sortedIdx, d_sortedIdx + N);
    // Count unique cell ids
    thrust::device_vector<int> uniqueKeys, counts;
    thrust::reduce_by_key(thrust::device, cellIdsVec.begin(), cellIdsVec.end(),
                          thrust::make_constant_iterator(1),
                          uniqueKeys.begin(), counts.begin());
    // Now we have counts per unique cell. We need to fill cellStart and cellCount.
    // We'll copy counts to host or use device-side scatter.
    // Since numCells may be large, we can do a host loop over unique keys.
    // Alternative: use thrust::scatter to fill cellCount.
    // But for simplicity, we copy to host and then back.
    std::vector<int> h_counts(numCells, 0);
    std::vector<int> h_keys = uniqueKeys; // size = number of non-empty cells
    // h_keys contains the cell ids that are non-empty.
    for (size_t i = 0; i < h_keys.size(); ++i) {
        h_counts[h_keys[i]] = counts[i];
    }
    cudaMemcpy(d_cellCount, h_counts.data(), numCells * sizeof(int), cudaMemcpyHostToDevice);

    // Compute cellStart: prefix sum
    std::vector<int> h_start(numCells);
    int running = 0;
    for (int i = 0; i < numCells; ++i) {
        h_start[i] = running;
        running += h_counts[i];
    }
    cudaMemcpy(d_cellStart, h_start.data(), numCells * sizeof(int), cudaMemcpyHostToDevice);

    // Cleanup cellIds
    cudaFree(d_cellIds);

    gridBuilt = true;
    return true;
}

bool UniformGridNeighbors::computeNeighbors(int k, std::vector<int>& outNeighbors) {
    if (!gridBuilt || N == 0) return false;
    outNeighbors.resize(N * k);

    // Process in chunks to limit memory usage per kernel
    const int CHUNK_SIZE = 2048; // number of queries per kernel launch
    // Allocate heap arrays for the chunk
    float* d_heapDist;
    int* d_heapIdx;
    int* d_out;
    cudaMalloc(&d_heapDist, CHUNK_SIZE * (k + 1) * sizeof(float));
    cudaMalloc(&d_heapIdx, CHUNK_SIZE * (k + 1) * sizeof(int));
    cudaMalloc(&d_out, CHUNK_SIZE * k * sizeof(int));

    for (int start = 0; start < N; start += CHUNK_SIZE) {
        int chunkSize = min(CHUNK_SIZE, N - start);
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
        // Copy output to host
        std::vector<int> h_out(chunkSize * k);
        cudaMemcpy(h_out.data(), d_out, chunkSize * k * sizeof(int), cudaMemcpyDeviceToHost);
        // Copy to final output
        for (int i = 0; i < chunkSize; ++i) {
            for (int j = 0; j < k; ++j) {
                outNeighbors[(start + i) * k + j] = h_out[i * k + j];
            }
        }
    }

    cudaFree(d_heapDist);
    cudaFree(d_heapIdx);
    cudaFree(d_out);
    return true;
}

// ... (freeDeviceData implementation)