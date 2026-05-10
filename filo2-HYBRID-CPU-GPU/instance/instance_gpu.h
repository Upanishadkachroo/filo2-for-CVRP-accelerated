#ifndef _FILO2_INSTANCE_GPU_H_
#define _FILO2_INSTANCE_GPU_H_

#include <cstddef>

struct CUstream_st;
using cuda_stream_t_fwd = CUstream_st*;

struct int2_fwd { int x, y; };

namespace cobra {

class Instance;   // forward declaration — avoids including Instance.hpp

extern float* d_xcoords;
extern float* d_ycoords;
extern int    d_n_vertices;

void instance_upload_coords(const Instance& inst,
                            cuda_stream_t_fwd stream = nullptr);

/// Release all device and pinned-host memory allocated by this module.
/// Safe to call even if upload was never called (no-op).
void instance_free_coords() noexcept;

/// Returns true if device coordinate arrays are allocated and populated.
bool instance_gpu_ready() noexcept;


void launch_batched_cost(const int2_fwd* d_pairs,
                         int*            d_out,
                         int             n_pairs,
                         cuda_stream_t_fwd stream = nullptr);

} 

#endif // _FILO2_INSTANCE_GPU_H_
