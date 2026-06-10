// base/gpu_utils.cuh
// Shared CUDA infrastructure included by all .cu translation units.
//
// Contents
//   1. CUDA_CHECK  — error-checking macro
//   2. gpu_device_info / gpu_available — device query helpers
//   3. pinned_alloc / pinned_free — page-locked host memory wrappers
//   4. extern declarations for the device coordinate arrays owned by Instance.cu

#pragma once

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstddef>

// ---------------------------------------------------------------------------
// 1. Error-checking macro
// ---------------------------------------------------------------------------

/// Wrap every CUDA API call with CUDA_CHECK to catch errors at the call site.
#define CUDA_CHECK(expr) \
    do { \
        cudaError_t _err = (expr); \
        if (_err != cudaSuccess) { \
            ::fprintf(stderr, \
                "[CUDA ERROR] %s:%d  →  %s\n", \
                __FILE__, __LINE__, \
                ::cudaGetErrorString(_err)); \
            ::std::exit(EXIT_FAILURE); \
        } \
    } while (0)

// ---------------------------------------------------------------------------
// 2. Device query helpers
// ---------------------------------------------------------------------------

/// Print device name, compute capability, SM count, and global memory.
/// Call once at program start before any kernel launch.
inline void gpu_device_info() {
    int device_count = 0;
    CUDA_CHECK(cudaGetDeviceCount(&device_count));
    if (device_count == 0) {
        ::fprintf(stderr, "[GPU] No CUDA devices found.\n");
        return;
    }
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    ::printf("[GPU] %s  |  SM %d.%d  |  %d SMs  |  %.1f GB\n",
             prop.name,
             prop.major, prop.minor,
             prop.multiProcessorCount,
             static_cast<double>(prop.totalGlobalMem) / (1 << 30));
}

/// Return true if a GPU with compute capability ≥ (minMajor, minMinor) exists.
inline bool gpu_available(int minMajor = 6, int minMinor = 0) {
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess || count == 0) return false;
    cudaDeviceProp prop;
    if (cudaGetDeviceProperties(&prop, 0) != cudaSuccess) return false;
    return (prop.major > minMajor) ||
           (prop.major == minMajor && prop.minor >= minMinor);
}

// ---------------------------------------------------------------------------
// 3. Pinned (page-locked) memory helpers
// ---------------------------------------------------------------------------
// cudaMemcpyAsync requires the host buffer to be pinned for true async
// overlap. Using these helpers instead of new[] / malloc avoids the hidden
// staging copy that CUDA inserts for non-pinned memory.

template<typename T>
inline T* pinned_alloc(::std::size_t n) {
    void* ptr = nullptr;
    CUDA_CHECK(cudaMallocHost(&ptr, n * sizeof(T)));
    return static_cast<T*>(ptr);
}

template<typename T>
inline void pinned_free(T* ptr) {
    if (ptr) CUDA_CHECK(cudaFreeHost(static_cast<void*>(ptr)));
}

// ---------------------------------------------------------------------------
// 4. Device coordinate arrays (defined in instance/Instance.cu)
// ---------------------------------------------------------------------------
// All CUDA translation units that need distance evaluations reference these
// via this header, avoiding duplicated extern declarations.

extern float* d_xcoords;   ///< Device copy of vertex x-coordinates (float)
extern float* d_ycoords;   ///< Device copy of vertex y-coordinates (float)
extern int    d_n_vertices; ///< Total number of vertices (customers + depot)
