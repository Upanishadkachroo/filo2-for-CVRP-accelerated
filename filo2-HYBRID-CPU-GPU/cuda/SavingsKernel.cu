#include "SavingsKernel.cuh"
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/scan.h>
#include <thrust/execution_policy.h>
#include <thrust/copy.h>
#include <cstdint>
#include <cmath>
#include <queue>
#include <cassert>

namespace cobra {

// ------------------------------------------------------------------
// Device: count valid neighbours (j > i) for each customer in chunk
// ------------------------------------------------------------------
__global__ void countValidNeighborsKernel(
    const int* __restrict__ neighbors,   // flat list, size N * maxNeighbors
    int N,
    int maxNeighbors,
    int* __restrict__ counts) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    const int* nbrs = neighbors + idx * maxNeighbors;
    int cnt = 0;
    for (int pos = 0; pos < maxNeighbors; ++pos) {
        int j = nbrs[pos];
        if (j > idx) ++cnt;   // only i < j
    }
    counts[idx] = cnt;
}

// ------------------------------------------------------------------
// Device: fill keys and values for valid (i,j) pairs
// ------------------------------------------------------------------
__global__ void fillSavingsKernel(
    const float* __restrict__ x,
    const float* __restrict__ y,
    const int* __restrict__ neighbors,
    int N,
    int maxNeighbors,
    const int* __restrict__ offsets,       // prefix sum of counts, size N+1
    float lambda,
    float* __restrict__ keys,              // output: -savings (for ascending sort)
    uint64_t* __restrict__ values) {       // packed (i<<32) | j

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    const int* nbrs = neighbors + idx * maxNeighbors;
    int off = offsets[idx];
    int pos = 0;
    float xi = x[idx], yi = y[idx];
    // depot is at (0,0) relative to depot (we shift coordinates)
    float d0i = sqrtf(xi*xi + yi*yi);

    for (int slot = 0; slot < maxNeighbors; ++slot) {
        int j = nbrs[slot];
        if (j > idx) {
            float xj = x[j], yj = y[j];
            float d0j = sqrtf(xj*xj + yj*yj);
            float dij = sqrtf((xi-xj)*(xi-xj) + (yi-yj)*(yi-yj));
            float sav = d0i + d0j - lambda * dij;
            keys[off + pos] = -sav;           // negate for ascending sort
            values[off + pos] = (uint64_t)idx << 32 | (uint32_t)j;
            ++pos;
        }
    }
}

// ------------------------------------------------------------------
// Host wrapper: compute and sort savings using GPU
// ------------------------------------------------------------------
bool computeSavingsGPU(const Instance& instance,
                       int neighbors_num,
                       double lambda,
                       std::vector<Saving>& savings) {

    const int N = instance.get_customers_num();
    if (N <= 0) return true;

    // Prepare host arrays: coordinates (relative to depot)
    const int depot = instance.get_depot();
    // Access the coordinate vectors directly (they are public in Instance)
    const auto& xcoords = instance.get_xcoords();
    const auto& ycoords = instance.get_ycoords();
    double depot_x = xcoords[depot];
    double depot_y = ycoords[depot];

    std::vector<float> h_x(N), h_y(N);
    for (int i = 0; i < N; ++i) {
        int cust = instance.get_customers_begin() + i;
        h_x[i] = (float)(xcoords[cust] - depot_x);
        h_y[i] = (float)(ycoords[cust] - depot_y);
    }

    // Build flat neighbor list: N * neighbors_num
    int maxNeighbors = neighbors_num;
    std::vector<int> h_neighbors(N * maxNeighbors, -1);
    for (int i = 0; i < N; ++i) {
        int cust = instance.get_customers_begin() + i;
        const auto& nbrs = instance.get_neighbors_of(cust);
        int copyCount = std::min((int)nbrs.size(), maxNeighbors);
        for (int p = 0; p < copyCount; ++p) {
            // Convert to 0‑based index
            h_neighbors[i * maxNeighbors + p] = nbrs[p] - instance.get_customers_begin();
        }
    }

    // Copy to device
    thrust::device_vector<float> d_x(h_x);
    thrust::device_vector<float> d_y(h_y);
    thrust::device_vector<int> d_neighbors(h_neighbors);
    float lambda_float = (float)lambda;

    // Process in chunks to limit memory usage.
    const int CHUNK_SIZE = 200000; // adjust based on GPU memory
    int chunkStart = 0;
    std::vector<std::vector<Saving>> sortedChunks;

    while (chunkStart < N) {
        int chunkEnd = std::min(chunkStart + CHUNK_SIZE, N);
        int chunkN = chunkEnd - chunkStart;

        // 1. Count valid neighbors per customer in this chunk
        thrust::device_vector<int> d_counts(chunkN);
        int blockSize = 256;
        int gridSize = (chunkN + blockSize - 1) / blockSize;
        countValidNeighborsKernel<<<gridSize, blockSize>>>(
            thrust::raw_pointer_cast(d_neighbors.data()) + chunkStart * maxNeighbors,
            chunkN, maxNeighbors,
            thrust::raw_pointer_cast(d_counts.data()));
        cudaDeviceSynchronize();

        // 2. Exclusive prefix sum to get offsets
        thrust::device_vector<int> d_offsets(chunkN + 1);
        thrust::exclusive_scan(thrust::device, d_counts.begin(), d_counts.end(), d_offsets.begin(), 0);
        int total = d_offsets[chunkN];

        if (total == 0) {
            chunkStart = chunkEnd;
            continue;
        }

        // 3. Allocate keys and values
        thrust::device_vector<float> d_keys(total);
        thrust::device_vector<uint64_t> d_values(total);

        // 4. Fill keys and values
        fillSavingsKernel<<<gridSize, blockSize>>>(
            thrust::raw_pointer_cast(d_x.data()) + chunkStart,
            thrust::raw_pointer_cast(d_y.data()) + chunkStart,
            thrust::raw_pointer_cast(d_neighbors.data()) + chunkStart * maxNeighbors,
            chunkN, maxNeighbors,
            thrust::raw_pointer_cast(d_offsets.data()),
            lambda_float,
            thrust::raw_pointer_cast(d_keys.data()),
            thrust::raw_pointer_cast(d_values.data()));
        cudaDeviceSynchronize();

        // 5. Sort by key ascending (so savings descending)
        thrust::sort_by_key(thrust::device, d_keys.begin(), d_keys.end(), d_values.begin());

        // 6. Copy back to host
        std::vector<float> h_keys(total);
        std::vector<uint64_t> h_values(total);
        thrust::copy(d_keys.begin(), d_keys.end(), h_keys.begin());
        thrust::copy(d_values.begin(), d_values.end(), h_values.begin());

        // 7. Convert to Saving structs (descending order)
        std::vector<Saving> chunkSavings;
        chunkSavings.reserve(total);
        for (int idx = 0; idx < total; ++idx) {
            float sav = -h_keys[idx];
            uint64_t packed = h_values[idx];
            int i = (int)(packed >> 32);
            int j = (int)(packed & 0xFFFFFFFF);
            int orig_i = instance.get_customers_begin() + i;
            int orig_j = instance.get_customers_begin() + j;
            chunkSavings.push_back({orig_i, orig_j, (double)sav});
        }
        sortedChunks.push_back(std::move(chunkSavings));

        chunkStart = chunkEnd;
    }

    if (sortedChunks.empty()) return false;

    // 8. Merge sorted chunks using priority queue (max‑heap on savings)
    struct Node {
        float value;
        int chunkId;
        int pos;
        bool operator<(const Node& other) const { return value < other.value; }
    };
    std::priority_queue<Node> pq;
    for (size_t c = 0; c < sortedChunks.size(); ++c) {
        if (!sortedChunks[c].empty()) {
            pq.push({(float)sortedChunks[c][0].value, (int)c, 0});
        }
    }

    savings.clear();
    savings.reserve(pq.size() * CHUNK_SIZE);
    while (!pq.empty()) {
        Node cur = pq.top(); pq.pop();
        int c = cur.chunkId;
        int p = cur.pos;
        savings.push_back(sortedChunks[c][p]);
        if (p + 1 < (int)sortedChunks[c].size()) {
            pq.push({(float)sortedChunks[c][p+1].value, c, p+1});
        }
    }

    return true;
}

} // namespace cobra
