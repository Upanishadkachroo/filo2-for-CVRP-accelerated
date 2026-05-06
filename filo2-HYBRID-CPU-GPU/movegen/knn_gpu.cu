// movegen/knn_gpu.cu

#include "../base/gpu_utils.cuh"
#include <cuda_runtime.h>
#include <float.h>

extern float* d_xcoords;
extern float* d_ycoords;

static constexpr int MAX_K = 64;
static constexpr int KNN_BLOCK = 256;

// ------------------------------------------------------------
// Block-level mutex
// ------------------------------------------------------------
__device__ int s_lock;

__device__ __forceinline__
void lock(int* mutex) {
    while (atomicCAS(mutex, 0, 1) != 0);
}

__device__ __forceinline__
void unlock(int* mutex) {
    atomicExch(mutex, 0);
}

// ------------------------------------------------------------
// Heap helpers
// ------------------------------------------------------------
__device__ __forceinline__
void heap_sift_down(float* hd, int* hi, int root, int size)
{
    while (true) {
        int largest = root;
        int l = 2*root+1, r = 2*root+2;

        if (l < size && hd[l] > hd[largest]) largest = l;
        if (r < size && hd[r] > hd[largest]) largest = r;

        if (largest == root) break;

        float td = hd[root]; hd[root] = hd[largest]; hd[largest] = td;
        int ti = hi[root]; hi[root] = hi[largest]; hi[largest] = ti;

        root = largest;
    }
}

__device__ __forceinline__
void heap_insert(float* hd, int* hi, int& size, int k, float dist, int idx)
{
    if (size < k) {
        hd[size] = dist;
        hi[size] = idx;
        size++;

        if (size == k) {
            for (int i = k/2 - 1; i >= 0; --i)
                heap_sift_down(hd, hi, i, k);
        }
    }
    else if (dist < hd[0]) {
        hd[0] = dist;
        hi[0] = idx;
        heap_sift_down(hd, hi, 0, k);
    }
}

// ------------------------------------------------------------
// Main kernel
// ------------------------------------------------------------
__global__ void knn_kernel(
    const float* __restrict__ xc,
    const float* __restrict__ yc,
    int n,
    int k,
    int* out_idx,
    float* out_dist)
{
    if (k > MAX_K) return;

    __shared__ float s_hd[MAX_K];
    __shared__ int   s_hi[MAX_K];
    __shared__ int   s_size;
    __shared__ int   s_mutex;

    int qi = blockIdx.x;
    if (qi >= n) return;

    // Init
    if (threadIdx.x == 0) {
        s_size = 0;
        s_mutex = 0;
        for (int i = 0; i < k; ++i) {
            s_hd[i] = FLT_MAX;
            s_hi[i] = -1;
        }
    }
    __syncthreads();

    float qx = xc[qi];
    float qy = yc[qi];

    // Sweep all points
    for (int j = threadIdx.x; j < n; j += blockDim.x) {

        if (j == qi) continue;

        float dx = qx - xc[j];
        float dy = qy - yc[j];
        float dist = dx*dx + dy*dy;  // no sqrt (faster)

        // Critical section
        lock(&s_mutex);
        heap_insert(s_hd, s_hi, s_size, k, dist, j);
        unlock(&s_mutex);
    }

    __syncthreads();

    // Write output
    if (threadIdx.x < k) {
        out_idx [qi * k + threadIdx.x] = s_hi[threadIdx.x];
        out_dist[qi * k + threadIdx.x] = s_hd[threadIdx.x];
    }
}

// ------------------------------------------------------------
// Partial kernel (FIXED)
// ------------------------------------------------------------
__global__ void knn_partial_kernel(
    const float* __restrict__ xc,
    const float* __restrict__ yc,
    const int* __restrict__ dirty,
    int n_dirty,
    int n,
    int k,
    int* out_idx,
    float* out_dist)
{
    if (k > MAX_K) return;

    __shared__ float s_hd[MAX_K];
    __shared__ int   s_hi[MAX_K];
    __shared__ int   s_size;
    __shared__ int   s_mutex;

    int bi = blockIdx.x;
    if (bi >= n_dirty) return;

    int qi = dirty[bi];

    if (threadIdx.x == 0) {
        s_size = 0;
        s_mutex = 0;
        for (int i = 0; i < k; ++i) {
            s_hd[i] = FLT_MAX;
            s_hi[i] = -1;
        }
    }
    __syncthreads();

    float qx = xc[qi];
    float qy = yc[qi];

    for (int j = threadIdx.x; j < n; j += blockDim.x) {

        if (j == qi) continue;

        float dx = qx - xc[j];
        float dy = qy - yc[j];
        float dist = dx*dx + dy*dy;

        lock(&s_mutex);
        heap_insert(s_hd, s_hi, s_size, k, dist, j);
        unlock(&s_mutex);
    }

    __syncthreads();

    if (threadIdx.x < k) {
        out_idx [bi * k + threadIdx.x] = s_hi[threadIdx.x];
        out_dist[bi * k + threadIdx.x] = s_hd[threadIdx.x];
    }
}

// ------------------------------------------------------------
// Host launchers
// ------------------------------------------------------------
void launch_knn_full(int n, int k,
                     int* h_knn_idx, float* h_knn_dist,
                     cudaStream_t stream)
{
    size_t bytes_i = (size_t)n * k * sizeof(int);
    size_t bytes_d = (size_t)n * k * sizeof(float);

    int* d_idx;
    float* d_dist;

    CUDA_CHECK(cudaMalloc(&d_idx, bytes_i));
    CUDA_CHECK(cudaMalloc(&d_dist, bytes_d));

    knn_kernel<<<n, KNN_BLOCK, 0, stream>>>(
        d_xcoords, d_ycoords, n, k, d_idx, d_dist);

    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaMemcpyAsync(h_knn_idx, d_idx, bytes_i,
                               cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaMemcpyAsync(h_knn_dist, d_dist, bytes_d,
                               cudaMemcpyDeviceToHost, stream));

    CUDA_CHECK(cudaStreamSynchronize(stream));

    CUDA_CHECK(cudaFree(d_idx));
    CUDA_CHECK(cudaFree(d_dist));
}

void launch_knn_partial(const int* h_dirty, int n_dirty,
                        int n, int k,
                        int* h_knn_idx, float* h_knn_dist,
                        cudaStream_t stream)
{
    if (n_dirty == 0) return;

    int* d_dirty;
    int* d_idx;
    float* d_dist;

    CUDA_CHECK(cudaMalloc(&d_dirty, n_dirty * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_idx, (size_t)n_dirty * k * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_dist, (size_t)n_dirty * k * sizeof(float)));

    CUDA_CHECK(cudaMemcpyAsync(d_dirty, h_dirty,
                               n_dirty * sizeof(int),
                               cudaMemcpyHostToDevice, stream));

    knn_partial_kernel<<<n_dirty, KNN_BLOCK, 0, stream>>>(
        d_xcoords, d_ycoords, d_dirty, n_dirty,
        n, k, d_idx, d_dist);

    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaMemcpyAsync(h_knn_idx, d_idx,
                               (size_t)n_dirty * k * sizeof(int),
                               cudaMemcpyDeviceToHost, stream));

    CUDA_CHECK(cudaMemcpyAsync(h_knn_dist, d_dist,
                               (size_t)n_dirty * k * sizeof(float),
                               cudaMemcpyDeviceToHost, stream));

    CUDA_CHECK(cudaStreamSynchronize(stream));

    CUDA_CHECK(cudaFree(d_dirty));
    CUDA_CHECK(cudaFree(d_idx));
    CUDA_CHECK(cudaFree(d_dist));
}