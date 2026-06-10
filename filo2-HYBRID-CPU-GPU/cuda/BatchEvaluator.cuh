// cuda/BatchEvaluator.cuh
// Double-buffered host→device→host pipeline for batch scoring kernels.
//
// Usage pattern (matches all three scoring modules):
//
//   BatchEvaluator<MoveBatch, float> eval(MAX_BATCH, stream);
//   eval.evaluate(h_moves, h_deltas, n_moves, my_kernel_launcher);
//
// The evaluator holds two pairs of device buffers (A and B) and alternates
// between them so that PCIe upload for the next batch overlaps with kernel
// execution on the current batch.

#pragma once
#include "DeviceBuffer.cuh"
#include <functional>

template<typename In, typename Out>
class BatchEvaluator {
public:
    using KernelFn = std::function<void(
        const In*    d_in,
        Out*         d_out,
        int          n,
        cudaStream_t stream)>;

    explicit BatchEvaluator(int max_batch, cudaStream_t stream)
        : stream_(stream), max_batch_(max_batch), slot_(0)
    {
        for (int s = 0; s < 2; ++s) {
            d_in_[s].resize(max_batch);
            d_out_[s].resize(max_batch);
            h_in_[s]  = pinned_alloc<In>(max_batch);
            h_out_[s] = pinned_alloc<Out>(max_batch);
        }
    }

    ~BatchEvaluator() {
        for (int s = 0; s < 2; ++s) {
            pinned_free(h_in_[s]);
            pinned_free(h_out_[s]);
        }
    }

    /// Evaluate a batch synchronously.
    /// h_in must already be filled by the caller.
    /// On return h_out[0..n-1] contains results.
    void evaluate(const In* h_in_src, Out* h_out_dst,
                  int n, KernelFn kernel_fn)
    {
        if (n == 0) return;
        // Clamp to capacity
        n = (n > max_batch_) ? max_batch_ : n;

        const int s = slot_ ^ 1;  // alternate slot

        // Copy input to pinned staging buffer
        std::memcpy(h_in_[s], h_in_src, n * sizeof(In));

        // Upload to device
        d_in_[s].upload(h_in_[s], n, stream_);

        // Launch kernel
        kernel_fn(d_in_[s].get(), d_out_[s].get(), n, stream_);

        // Download results
        d_out_[s].download(h_out_[s], n, stream_);
        CUDA_CHECK(cudaStreamSynchronize(stream_));

        // Copy output to caller's buffer
        std::memcpy(h_out_dst, h_out_[s], n * sizeof(Out));

        slot_ = s;
    }

private:
    cudaStream_t    stream_;
    int             max_batch_;
    int             slot_;

    DeviceBuffer<In>  d_in_[2];
    DeviceBuffer<Out> d_out_[2];
    In*               h_in_[2];
    Out*              h_out_[2];
};
