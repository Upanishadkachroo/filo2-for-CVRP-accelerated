# Move Generators & GPU k-NN Acceleration (FILO2 – CVRP)

## Overview

This module implements a **high-performance move generation framework** for large-scale **Capacitated Vehicle Routing Problems (CVRP)**, inspired by the FILO2 algorithm.

It combines:

* **CPU-based Move Generators** (incremental local search support)
* **GPU-accelerated k-Nearest Neighbor (k-NN) graph construction**

The goal is to efficiently generate a **restricted candidate neighborhood** that enables fast and scalable local search — even for **million-node instances**.

---

## Core Idea

Instead of evaluating all possible edges (which is (O(n^2))), we:

1. Build a **k-NN graph** (each node connects to its k closest neighbors)
2. Generate **move candidates only from this graph**
3. Dynamically activate/deactivate subsets of moves during search

This reduces complexity to approximately **(O(n \cdot k))**.

---

## Module Structure

```
movegen/
│
├── MoveGenerators.hpp      # CPU move generator + heap logic
├── knn_gpu.cu              # GPU-based k-NN construction
└── README.md
```

---

# CPU Component: MoveGenerators

## Purpose

Manages **candidate moves** used in local search heuristics.

Each move represents a directed edge:

```
(i → j)
```

For every undirected edge, we store:

```
(i → j) and (j → i)
```

---

## Key Components

### 1. `MoveGenerator`

Represents a single move:

* `i, j` → endpoints
* `delta` → cost improvement
* `heap_index` → position in priority queue
* `computed_for_ejch` → optimization flag

---

### 2. `MoveGeneratorsHeap`

A specialized **binary heap** used to:

* Retrieve best improving move
* Update move priorities dynamically

Supports:

* `insert`
* `remove`
* `update (change_value)`
* `get (best move)`

---

### 3. Move Storage Design

Moves are stored in a **paired layout**:

```
Index:   0     1     2     3
        (i,j) (j,i) (a,b) (b,a)
```

#### Benefits:

* Constant-time twin lookup:

```cpp
idx ^ 1
```

* Base move index:

```cpp
idx & ~1
```

---

### 4. Active Move Filtering

To reduce computation:

* Each vertex activates only a **subset of its k neighbors**
* Controlled via:

```cpp
set_active_percentage(...)
```

This enables:

* Adaptive neighborhood size
* Faster local search iterations

---

### 5. Incremental Update Mechanism

Efficient updates using:

* **Timestamp system** → avoids recomputation
* **Update bits (Flat2DVector)** → track affected edges
* **Sparse sets** → fast unique vertex tracking

---

## Workflow (CPU Side)

```
k-NN graph → build base moves
          → activate subset
          → push into heap
          → local search uses heap
          → update affected moves only
```

---

# GPU Component: k-NN Graph Construction

## Purpose

Efficiently compute **k nearest neighbors for each vertex** using CUDA.

---

## Why GPU?

CPU complexity:

```
O(n × k) distance computations
```

For:

```
n = 1,000,000, k = 20 → 20M distance evaluations
```

GPU parallelizes across all vertices:

* One block per query point
* Threads cooperatively scan all candidates

---

## Kernel Design

### Thread Mapping

| Level   | Role                        |
| ------- | --------------------------- |
| Block   | One query vertex (i)        |
| Threads | Scan all candidate vertices |

---

### Shared Memory Heap

Each block maintains:

```cpp
__shared__ float s_hd[MAX_K]; // distances
__shared__ int   s_hi[MAX_K]; // indices
```

This is a **fixed-size max-heap** of size `k`.

---

### Heap Logic

* If heap not full → insert directly
* If full → replace max element if better candidate found

---

### Synchronization Strategy

* Block-level mutex using `atomicCAS`
* Ensures safe concurrent updates to shared heap

---

### Distance Optimization

We use:

```cpp
dx*dx + dy*dy
```

instead of:

```cpp
sqrt(dx*dx + dy*dy)
```

Avoids expensive square root (ordering preserved)

---

## Kernels

### 1. `knn_kernel`

* Computes full k-NN graph
* One block per vertex

---

### 2. `knn_partial_kernel`

* Processes only **dirty vertices**
* Used for incremental updates (gamma-triggered refresh)

---

## Workflow (GPU Side)

```
Input: coordinates (x, y)
      ↓
Launch kernel (n blocks)
      ↓
Each block builds local k-NN heap
      ↓
Write results to global memory
      ↓
Copy back to CPU
```

---

# CPU ↔ GPU Integration

### Pipeline

```
GPU k-NN
   ↓
Neighbor lists (k per vertex)
   ↓
MoveGenerators builds moves
   ↓
Heap-based local search
```

---

## Performance Benefits

| Component    | CPU Only                | GPU Accelerated |
| ------------ | ----------------------- | --------------- |
| k-NN Build   | Seconds                 | ~50 ms          |
| Scalability  | Limited                 | Million nodes   |
| Local Search | Faster (due to pruning) | Same            |

---

# Design Trade-offs

### Pros

* Massive speedup for large instances
* Reduced search space
* Incremental updates supported
* Cache-friendly structures

---

### Cons

* GPU heap uses locking → not optimal
* Memory overhead for move storage
* k-NN approximation limits search space

---

# Future Optimizations

### High-impact upgrades

1. **Warp-level heap (lock-free)**
2. **Parallel reduction instead of mutex**
3. **FAISS-style approximate k-NN**
4. **GPU → CPU zero-copy integration**
5. **Shared memory tiling for coalesced access**

---

# Use Case: FILO2 for CVRP

This module enables:

* Efficient neighborhood generation
* Fast move evaluation
* Scalable local search

Critical for solving:

* Large-scale CVRP (100K – 1M customers)
* Real-time routing problems

---

# Summary

This implementation provides:

* Efficient **move generation engine**
* GPU-accelerated **k-NN construction**
* Incremental update mechanisms
* Scalable architecture for large instances

---
