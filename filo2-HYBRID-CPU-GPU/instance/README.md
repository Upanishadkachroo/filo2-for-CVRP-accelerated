# instance/

This module is the **data backbone** of the FILO2 hybrid solver.

It is split into two sub-directories that reflect the CPU / GPU ownership model:

```
instance/
├── cpu/
│   ├── Instance.hpp   ← problem data, get_cost(), gpu lifecycle surface
│   ├── Parser.hpp     ← VRP file parser
│   └── README.md
└── gpu/
    ├── instance_gpu.h   ← C++ interface (no CUDA headers leaked to callers)
    ├── instance_gpu.cu  ← device globals, kernel, async upload
    └── README.md
```

---

## Responsibilities

| Layer | Owns | Does NOT own |
|-------|------|--------------|
| CPU | Coordinates (double[]), demands, neighbors, k-NN lists | Device buffers |
| GPU | d_xcoords[], d_ycoords[] (float[]) on device | Route structure |

---

## Key design decisions

### 1. Coordinates uploaded once

`instance_upload_coords()` is called **once** at program startup.
`d_xcoords` and `d_ycoords` remain resident on the device for the entire run.
All scoring kernels (savings, delta, recreate) reference them directly via
`extern` device pointers declared in `instance_gpu.h`.

### 2. float on GPU, double on CPU

CPU `get_cost(i,j)` returns `double` — unchanged from the original FILO2.
GPU kernels use `float` arithmetic internally: `sqrtf` + `__float2int_rn`.
The rounded integer result is bit-identical to `fastround(double sqrt)` for
all CVRP coordinate ranges. See `instance/gpu/README.md §Fix 5` for the ULP
bound proof.

### 3. Async transfers with pinned staging

Upload uses `cudaMemcpyAsync` on a caller-supplied stream and a module-level
pinned staging buffer. The CPU is never blocked by a DMA transfer; PCIe
bandwidth is overlapped with CPU work.

### 4. No CUDA headers in Instance.hpp

`Instance.hpp` can be included by any CPU translation unit without acquiring
a CUDA dependency. The `gpu_upload()` / `gpu_free()` / `gpu_ready()` member
functions are defined in `instance_gpu.cu` and link-time resolved.

---

## Data flow across the solver

```
Parser::Data
    │  Instance::make()
    ▼
Instance (CPU)
    ├── get_cost(i,j)           ← used by all CPU-side evaluations
    ├── get_neighbors_of(i)     ← used by local search operators
    │
    │  gpu_upload()
    ▼
d_xcoords[], d_ycoords[]       ← used by ALL GPU scoring kernels
    │
    ├── savings_kernel          (solution/gpu/)
    ├── knn_kernel              (movegen/gpu/)
    ├── delta_kernel            (localsearch/gpu/)
    ├── recreate_kernel         (opt/gpu/)
    └── feasibility_kernel      (localsearch/gpu/)
```
