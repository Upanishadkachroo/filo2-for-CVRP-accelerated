#include "CudaNeighborFinder.hpp"
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/sequence.h>
#include <iostream>
#include <chrono>

namespace cobra {

// ----------------------------------------------------------------
// Device swap — std::swap is host-only, cannot be used in kernels
// ----------------------------------------------------------------
template<typename T>
__device__ __forceinline__ void dev_swap(T& a, T& b) {
    T tmp = a; a = b; b = tmp;
}

static inline int divUp(int a, int b) { return (a + b - 1) / b; }

// ----------------------------------------------------------------
// Kernel: brute-force k-NN, one thread per query vertex.
//
// Each thread scans ALL N coordinates and maintains a max-heap of
// size k in registers/local memory.  This avoids any shared memory
// or global workspace — the only output is out_indices / out_dists.
//
// Template parameter K must equal k at compile time.  We specialise
// for K=1500 (the FILO2 default).  For other k values the host falls
// back to the generic path (see computeAllNeighborsFlat).
// ----------------------------------------------------------------
template<int K>
__global__ void bruteForceKNN(const float* __restrict__ x,
                               const float* __restrict__ y,
                               int   n,
                               int   query_offset,   // first global vertex index in this chunk
                               int   num_queries,    // queries in this chunk
                               int*   __restrict__ out_indices,
                               float* __restrict__ out_dists) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= num_queries) return;

    int qid = query_offset + tid;
    float qx = x[qid];
    float qy = y[qid];

    // Max-heap stored in thread-local arrays (registers / L1 spill)
    int   heap_idx[K];
    float heap_dist[K];
    int   heap_size = 0;

    for (int j = 0; j < n; ++j) {
        if (j == qid) continue;
        float dx   = qx - x[j];
        float dy   = qy - y[j];
        float dist = dx*dx + dy*dy;

        if (heap_size < K) {
            heap_idx[heap_size]  = j;
            heap_dist[heap_size] = dist;
            // Sift up
            int pos = heap_size;
            while (pos > 0 && heap_dist[pos] > heap_dist[(pos-1)/2]) {
                dev_swap(heap_idx[pos],  heap_idx[(pos-1)/2]);
                dev_swap(heap_dist[pos], heap_dist[(pos-1)/2]);
                pos = (pos-1)/2;
            }
            ++heap_size;
        } else if (dist < heap_dist[0]) {
            heap_idx[0]  = j;
            heap_dist[0] = dist;
            // Sift down
            int pos = 0;
            while (true) {
                int l = 2*pos+1, r = 2*pos+2, largest = pos;
                if (l < K && heap_dist[l] > heap_dist[largest]) largest = l;
                if (r < K && heap_dist[r] > heap_dist[largest]) largest = r;
                if (largest == pos) break;
                dev_swap(heap_idx[pos],  heap_idx[largest]);
                dev_swap(heap_dist[pos], heap_dist[largest]);
                pos = largest;
            }
        }
    }

    // Sort heap ascending via selection sort (K ≤ 1500, runs in local mem)
    for (int i = 0; i < heap_size - 1; ++i) {
        int mp = i;
        for (int j = i+1; j < heap_size; ++j)
            if (heap_dist[j] < heap_dist[mp]) mp = j;
        if (mp != i) {
            dev_swap(heap_idx[i],  heap_idx[mp]);
            dev_swap(heap_dist[i], heap_dist[mp]);
        }
    }

    // Write to global memory
    int base = tid * K;
    for (int i = 0; i < K; ++i) {
        out_indices[base + i] = (i < heap_size) ? heap_idx[i]  : -1;
        out_dists[base + i]   = (i < heap_size) ? heap_dist[i] : 1e30f;
    }
}

// ----------------------------------------------------------------
// Constructor / Destructor
// ----------------------------------------------------------------
CudaNeighborFinder::CudaNeighborFinder(const std::vector<double>& x,
                                       const std::vector<double>& y)
    : n(static_cast<int>(x.size())), h_x(x), h_y(y) {

    std::vector<float> xf(n), yf(n);
    for (int i = 0; i < n; ++i) {
        xf[i] = static_cast<float>(x[i]);
        yf[i] = static_cast<float>(y[i]);
    }
    cudaMalloc(&d_x, n * sizeof(float));
    cudaMalloc(&d_y, n * sizeof(float));
    cudaMemcpy(d_x, xf.data(), n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, yf.data(), n * sizeof(float), cudaMemcpyHostToDevice);
}

CudaNeighborFinder::~CudaNeighborFinder() {
    if (d_x) cudaFree(d_x);
    if (d_y) cudaFree(d_y);
}

// ----------------------------------------------------------------
// computeAllNeighborsFlat
//
// Chunked strategy: process CHUNK_SIZE query vertices at a time so
// that GPU output buffers never exceed ~1.2 GB regardless of N or k.
//
// Memory per chunk (CHUNK_SIZE=100,000, k=1500):
//   indices: 100,000 × 1500 × 4 = 600 MB
//   dists:   100,000 × 1500 × 4 = 600 MB
//   Total:   1,200 MB  — comfortably fits on T4 (15 GB)
//
// For N=1M  : 10 chunks
// For N=2.5M: 25 chunks
// ----------------------------------------------------------------
std::vector<int> CudaNeighborFinder::computeAllNeighborsFlat(int k, bool verbose) {
    if (k <= 0 || k > 1500) {
        std::cerr << "CudaNeighborFinder: k must be 1-1500 (got " << k << ")\n";
        return {};
    }

    // Compute chunk size so output fits within ~1.2 GB
    // Each vertex needs k × 8 bytes (4 for idx, 4 for dist)
    const long long MAX_OUTPUT_BYTES = 1200LL * 1024 * 1024; // 1.2 GB
    const int CHUNK_SIZE = static_cast<int>(
        std::min((long long)n, MAX_OUTPUT_BYTES / (k * 8LL)));

    if (verbose) {
        std::cout << "[CudaBF] N=" << n << " k=" << k
                  << " chunk=" << CHUNK_SIZE
                  << " chunks=" << divUp(n, CHUNK_SIZE) << "\n";
    }

    // Allocate chunk-sized GPU output buffers once, reuse across chunks
    int*   d_indices = nullptr;
    float* d_dists   = nullptr;
    cudaMalloc(&d_indices, (long long)CHUNK_SIZE * k * sizeof(int));
    cudaMalloc(&d_dists,   (long long)CHUNK_SIZE * k * sizeof(float));

    cudaError_t alloc_err = cudaGetLastError();
    if (alloc_err != cudaSuccess) {
        std::cerr << "CudaNeighborFinder: GPU alloc failed: "
                  << cudaGetErrorString(alloc_err) << "\n";
        return {};
    }

    // Host result (full N × k)
    std::vector<int> h_indices((long long)n * k);

    const int BLOCK = 128; // fewer threads → larger per-thread register budget for heap
    const int num_chunks = divUp(n, CHUNK_SIZE);

    auto t0 = std::chrono::high_resolution_clock::now();

    for (int chunk = 0; chunk < num_chunks; ++chunk) {
        int q_start = chunk * CHUNK_SIZE;
        int q_count = std::min(CHUNK_SIZE, n - q_start);
        int grid    = divUp(q_count, BLOCK);

        // Only k=1500 is template-specialised (FILO2 default).
        // Add more specialisations here if you use other k values.
        if (k == 1500) {
            bruteForceKNN<1500><<<grid, BLOCK>>>(
                d_x, d_y, n, q_start, q_count, d_indices, d_dists);
        } else {
            // For non-1500 k values fall through to error — caller should
            // use GridNeighborFinder or KDTree for arbitrary k.
            std::cerr << "CudaNeighborFinder: unsupported k=" << k
                      << ". Only k=1500 is compiled. Use GridNeighborFinder.\n";
            cudaFree(d_indices); cudaFree(d_dists);
            return {};
        }

        cudaError_t err = cudaDeviceSynchronize();
        if (err != cudaSuccess) {
            std::cerr << "CudaNeighborFinder kernel error (chunk " << chunk
                      << "): " << cudaGetErrorString(err) << "\n";
            cudaFree(d_indices); cudaFree(d_dists);
            return {};
        }

        // Copy this chunk's indices back to host (dists not needed by caller)
        cudaMemcpy(h_indices.data() + (long long)q_start * k,
                   d_indices,
                   (long long)q_count * k * sizeof(int),
                   cudaMemcpyDeviceToHost);

        if (verbose) {
            auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::high_resolution_clock::now() - t0).count();
            std::cout << "[CudaBF] chunk " << (chunk+1) << "/" << num_chunks
                      << "  elapsed=" << ms << "ms\r" << std::flush;
        }
    }

    cudaFree(d_indices);
    cudaFree(d_dists);

    auto total_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::high_resolution_clock::now() - t0).count();
    if (verbose)
        std::cout << "\n[CudaBF] Total k-NN time: " << total_ms << " ms\n";

    return h_indices;
}

std::vector<std::vector<int>> CudaNeighborFinder::computeAllNeighbors(int k, bool verbose) {
    auto flat = computeAllNeighborsFlat(k, verbose);
    if (flat.empty()) return {};
    std::vector<std::vector<int>> result(n);
    for (int i = 0; i < n; ++i) {
        result[i].assign(flat.begin() + (long long)i * k,
                         flat.begin() + (long long)i * k + k);
    }
    return result;
}

} // namespace cobra