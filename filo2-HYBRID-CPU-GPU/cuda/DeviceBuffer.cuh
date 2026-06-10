// cuda/DeviceBuffer.cuh
// RAII wrapper for device memory — prevents leaks and simplifies kernel code.

#pragma once
#include "../base/gpu_utils.cuh"
#include <cstddef>
#include <utility>    // std::exchange

/// Owning handle for a device-memory array of type T.
/// Analogous to std::unique_ptr<T[]> but for GPU memory.
template<typename T>
class DeviceBuffer {
public:
    /// Allocate n elements on the device.
    explicit DeviceBuffer(std::size_t n = 0) : ptr_(nullptr), n_(n) {
        if (n > 0) CUDA_CHECK(cudaMalloc(&ptr_, n * sizeof(T)));
    }

    /// Move constructor — transfers ownership.
    DeviceBuffer(DeviceBuffer&& o) noexcept
        : ptr_(std::exchange(o.ptr_, nullptr)),
          n_  (std::exchange(o.n_,   0)) {}

    /// Move assignment.
    DeviceBuffer& operator=(DeviceBuffer&& o) noexcept {
        if (this != &o) {
            free();
            ptr_ = std::exchange(o.ptr_, nullptr);
            n_   = std::exchange(o.n_,   0);
        }
        return *this;
    }

    // Non-copyable
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    ~DeviceBuffer() { free(); }

    /// Resize — frees existing allocation if size changes.
    void resize(std::size_t new_n) {
        if (new_n == n_) return;
        free();
        n_ = new_n;
        if (n_ > 0) CUDA_CHECK(cudaMalloc(&ptr_, n_ * sizeof(T)));
    }

    T*          get()  const { return ptr_; }
    std::size_t size() const { return n_; }

    /// Upload n elements from host pointer h (synchronous).
    void upload(const T* h, std::size_t n, cudaStream_t s = 0) {
        CUDA_CHECK(cudaMemcpyAsync(ptr_, h, n * sizeof(T),
                                   cudaMemcpyHostToDevice, s));
    }

    /// Download n elements to host pointer h (synchronous).
    void download(T* h, std::size_t n, cudaStream_t s = 0) const {
        CUDA_CHECK(cudaMemcpyAsync(h, ptr_, n * sizeof(T),
                                   cudaMemcpyDeviceToHost, s));
    }

private:
    void free() {
        if (ptr_) { CUDA_CHECK(cudaFree(ptr_)); ptr_ = nullptr; n_ = 0; }
    }

    T*          ptr_;
    std::size_t n_;
};
