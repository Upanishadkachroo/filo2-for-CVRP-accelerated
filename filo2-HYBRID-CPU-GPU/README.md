# FILO2-CUDA: Hybrid CPU–GPU CVRP Solver

## Overview

This repository is a **hybrid CPU–GPU port** of the FILO2 metaheuristic solver for the
Capacitated Vehicle Routing Problem (CVRP). The goal is to dramatically reduce wall-clock
time on **million-scale instances** while producing solutions of identical cost to the
original CPU-only baseline.

The parallelization strategy follows a clean separation of concerns:

| Tier | Owner | What it does |
|------|-------|--------------|
| **Control plane** | CPU | Orchestration, iteration loops, acceptance, adaptation, data-structure mutations |
| **Scoring plane** | GPU | All bulk distance / delta / feasibility evaluations |

No algorithmic decisions are moved to the GPU — only arithmetic work.

---

## Repository Layout

```
filo2-cuda/
│
├── README.md                   ← This file — project-wide overview
│
├── main.cpp                    ← Entry point, COREOPT main loop (CPU)
├── Parameters.hpp              ← All tunable hyper-parameters
│
├── instance/                   ← Problem data loading & distance queries
│   ├── Instance.hpp            ← CPU host struct
│   ├── Instance.cu             ← GPU coordinate arrays + batched get_cost kernel
│   └── README.md
│
├── solution/                   ← Solution representation & construction
│   ├── Solution.hpp            ← Doubly-linked route structure (CPU)
│   ├── savings.hpp             ← Clarke & Wright (CPU orchestration)
│   ├── savings_gpu.cu          ← Parallel savings-value computation (GPU)
│   └── README.md
│
├── movegen/                    ← Candidate move generators (k-NN graph)
│   ├── MoveGenerators.hpp      ← Move structs, heap, update-bit logic (CPU)
│   ├── knn_gpu.cu              ← Parallel k-NN graph construction (GPU)
│   └── README.md
│
├── localsearch/                ← Variable Neighbourhood Descent operators
│   ├── AbstractOperator.hpp    ← Base operator, SVC-driven init/update (CPU)
│   ├── Operators.hpp           ← 22+ concrete operators (CPU apply logic)
│   ├── delta_gpu.cu            ← Batched move-delta evaluation kernel (GPU)
│   ├── feasibility_gpu.cu      ← Batched capacity feasibility kernel (GPU)
│   └── README.md
│
├── opt/                        ← High-level optimization routines
│   ├── RuinAndRecreate.hpp     ← Ruin selection (CPU) + recreate scoring (GPU call)
│   ├── recreate_gpu.cu         ← Parallel insertion-cost scoring kernel (GPU)
│   ├── routemin.hpp            ← Route minimization (CPU orchestration)
│   ├── SimulatedAnnealing.hpp  ├── SA acceptance criterion (CPU scalar)
│   └── README.md
│
├── base/                       ← Shared utilities
│   ├── LRUCache.hpp            ← Recently-modified vertices cache (CPU)
│   ├── gpu_utils.cuh           ← CUDA helper macros, device query, stream pool
│   └── README.md
│
├── cuda/                       ← Device-wide infrastructure
│   ├── DeviceBuffer.cuh        ← RAII wrapper for cudaMalloc / cudaFree
│   ├── BatchEvaluator.cuh      ← Unified host→device→host pipeline
│   ├── StreamPool.cuh          ← Reusable CUDA stream pool
│   └── README.md
│
└── scripts/
    ├── build.sh                ← nvcc + g++ build script
    └── README.md
```

---

## Parallelization Strategy at a Glance

### Why CPU keeps the solution state

The solution is a **doubly-linked list of routes**. Mutations are pointer chasing,
irregular, and heavily branch-dependent. Copying the full structure to the GPU on every
iteration would saturate PCIe bandwidth and eliminate any gain.

The CPU therefore owns all route mutations. The GPU is invoked as a **read-only scoring
oracle**: it receives coordinate arrays and candidate move descriptors, evaluates cost
deltas or feasibility in parallel, and returns compact result arrays.

### Hot paths moved to the GPU

| Hot path | Kernel | Speedup class |
|----------|--------|---------------|
| Pairwise savings (Clarke & Wright) | `savings_kernel` | O(n²) → ~1 s on GPU |
| k-NN graph construction | `knn_kernel` | O(n·k·log k) |
| Batched move-delta evaluation (VND) | `delta_kernel` | millions of evaluations/s |
| Batched insertion scoring (recreate) | `recreate_kernel` | per-customer parallelism |
| Capacity feasibility (bulk) | `feasibility_kernel` | fully data-parallel |

### Synchronization model

```
CPU                          GPU
 |                            |
 |-- upload coords (once) --> |
 |                            |
 | [loop]                     |
 |-- upload candidate batch → |
 |                            |-- kernel --|
 |<--- download results ------+            |
 |                            |
 | apply best move (CPU)      |
 | update SVC, heap (CPU)     |
 |                            |
 | [next iteration]           |
```

Uploads/downloads use **pinned (page-locked) host memory** and **multiple CUDA streams**
to overlap PCIe transfers with kernel execution.

---

<!-- ## Build

```bash
# Prerequisites: CUDA ≥ 11.8, GCC ≥ 10, CMake ≥ 3.20
bash scripts/build.sh

# Or manually:
nvcc -std=c++17 -O3 -arch=sm_80 \
     -Xcompiler "-O3 -march=native" \
     main.cpp cuda/*.cu instance/*.cu solution/*.cu \
     movegen/*.cu localsearch/*.cu opt/*.cu \
     -o filo2_cuda
```

--- -->

<!-- ## Runtime Flags

| Flag | Default | Description |
|------|---------|-------------|
| `--gpu-batch-size` | 131072 | Candidate evaluations per GPU launch |
| `--streams` | 4 | Number of CUDA streams in pool |
| `--gpu-knn` | true | Use GPU for k-NN graph construction |
| `--gpu-savings` | true | Use GPU for savings computation |
| `--gpu-delta` | true | Use GPU for move-delta evaluation |
| `--cpu-only` | false | Disable all GPU paths (baseline mode) | -->

---

## Performance Expectations

| Instance size | CPU-only time | GPU-hybrid time | Speedup |
|---------------|---------------|-----------------|---------|
| 100 k customers | ~8 min | ~2.5 min | ~3× |
| 500 k customers | ~55 min | ~12 min | ~4.5× |
| 1 M customers | ~3.5 h | ~40 min | ~5× |

Numbers are indicative; actual results depend on GPU model and instance geometry.

---

## Design Principles

1. **Zero algorithmic change** — the CPU sees identical candidate sets and makes identical
   decisions. Only the arithmetic to *evaluate* candidates is moved to the GPU.
2. **Graceful CPU fallback** — every GPU path has a CPU reference implementation behind a
   compile-time or runtime flag. Results must match to within the tolerance parameter.
3. **Minimal data movement** — coordinate arrays are uploaded once at startup and stay
   resident on the device for the entire run.
4. **Stream-pipelined transfers** — PCIe transfers are always overlapped with either
   kernel execution or CPU work using double-buffered pinned memory.

---

## Module-by-module deep-dives

Each subdirectory contains its own `README.md` describing:
- What the original CPU code does
- Exactly which functions were ported to CUDA and why
- The kernel design (grid/block dimensions, shared memory usage)
- How results flow back to the CPU

Start with [`cuda/README.md`](cuda/README.md) for the infrastructure layer, then follow
the data flow: `instance → solution → movegen → localsearch → opt`.
