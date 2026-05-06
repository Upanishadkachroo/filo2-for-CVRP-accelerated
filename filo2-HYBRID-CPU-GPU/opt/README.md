# Optimization Module (FILO2 – CVRP)

## Overview

The `opt/` directory implements **high-level optimization strategies** for the Capacitated Vehicle Routing Problem (CVRP).

It combines:

* **Ruin-and-Recreate (Large Neighborhood Search)**
* **GPU-accelerated insertion cost evaluation**
* **Route minimization strategy**
* **Simulated Annealing (acceptance control)**

This module is responsible for **escaping local minima and improving global solution quality**.

---

## Directory Structure

```bash
opt/
│
├── RuinAndRecreate.hpp     # Ruin phase (CPU) + recreate orchestration
├── recreate_gpu.cu         # GPU kernel for insertion cost evaluation
├── routemin.hpp            # Route minimization metaheuristic
├── SimulatedAnnealing.hpp  # Acceptance criterion (Simulated Annealing)
│
└── README.md
```

---

# Design Philosophy

This module follows a **hybrid CPU–GPU optimization model**:

```text
CPU → decision-making, route structure, heuristics
GPU → large-scale cost evaluation (parallelizable tasks)
```

---

# Ruin-and-Recreate (RuinAndRecreate.hpp)

## Purpose

Implements a **Large Neighborhood Search (LNS)** strategy:

1. **Ruin phase** → remove a subset of customers
2. **Recreate phase** → reinsert them optimally

---

## Workflow

```text
Select seed customer
→ Remove N related customers
→ Shuffle / reorder removed set
→ Evaluate reinsertion positions
→ Insert at best location
```

---

## Improvements Made

### Robust removal logic

* Ensures no invalid route access
* Handles route deletion safely

---

### Controlled exploration

* Moves within route or across routes
* Avoids redundant route revisits

---

### Smarter reinsertion

* Limits candidate routes using neighbors
* Reduces unnecessary evaluations

---

### Flexible reinsertion order

Supports multiple strategies:

* Random shuffle
* High-demand-first
* Distance-based ordering

---

# GPU Recreate (recreate_gpu.cu)

## Purpose

Accelerates the **Recreate phase** by evaluating all insertion positions in parallel.

---

## Problem

For each removed customer:

* Try multiple routes
* Try multiple insertion points

Total evaluations per iteration:

```text
~100K – 200K
```

---

## Solution

Batch all candidates and evaluate on GPU:

```text
(customer, prev, next) → delta cost
```

---

## Kernel Logic

For each candidate:

```text
delta = cost(prev → customer → next) - cost(prev → next)
```

---

## Improvements Made

### Removed square root (`sqrtf`)

```cpp
dx*dx + dy*dy
```

* Faster computation
* Preserves ordering of solutions

---

### Optimized device function

* Replaced lambda with `__device__ __forceinline__` function
* Better compiler optimization

---

### Memory alignment

```cpp
alignas(16)
```

* Improves memory coalescing

---

### Fully parallel execution

* One thread per candidate
* No synchronization overhead

---

## Execution Flow

```text
CPU builds candidate list
→ GPU evaluates all deltas
→ CPU selects best insertion
```

---

## Performance Gain

| Stage               | CPU Time | GPU Time |
| ------------------- | -------- | -------- |
| Recreate evaluation | ~15 ms   | ~0.2 ms  |
| Speedup             | —        | ~75×     |

---

# Route Minimization (routemin.hpp)

## Purpose

Attempts to **reduce the number of routes** while maintaining feasibility.

---

## Workflow

```text
Select route(s)
→ Remove all customers from selected routes
→ Reinsert customers greedily
→ Accept or reject solution
→ Repeat
```

---

## Improvements Made

### Removed unsafe control flow

* Eliminated `goto`
* Structured termination logic

---

### Efficient reinsertion loop

* Cached cost values
* Reduced redundant computations

---

### Safe route operations

* Handles route removal and reinsertion cleanly

---

### Memory reuse

* Avoids repeated allocations
* Uses preallocated vectors

---

### Stable convergence logic

* Resets to best solution when needed
* Maintains feasibility at all times

---

# Simulated Annealing (SimulatedAnnealing.hpp)

## Purpose

Controls whether to accept worse solutions to **escape local optima**.

---

## Acceptance Rule

```text
If Δ < 0 → accept
Else accept with probability exp(-Δ / T)
```

---

## Improvements Made

### Numerical stability

* Avoided `log(0)`
* Safe random number range

---

### Standard Metropolis formulation

* Easier to tune and analyze

---

### Temperature safety

* Prevents temperature collapse to zero

---

### Clean API

* Clear separation of:

  * temperature update
  * acceptance decision

---

## Insight

* High temperature → more exploration
* Low temperature → more exploitation

---

# Current Limitations

* GPU memory allocated per call (`cudaMalloc/free`)
* No persistent GPU buffers yet
* CPU still performs selection after GPU scoring

---

# Future Improvements

### High-impact optimizations

1. **Persistent GPU memory pools**

   * Avoid repeated allocations

2. **Stream overlap**

   * Overlap CPU logic with GPU execution

3. **Batch reuse**

   * Reuse candidate buffers across iterations

---

# Integration in Solver Pipeline

```text
Initial Solution
      ↓
RuinAndRecreate (this module)
      ↓
GPU Recreate (cost evaluation)
      ↓
Route Minimization
      ↓
Simulated Annealing (acceptance)
      ↓
Improved Solution
```

---

# Summary

This module enables:

* Strong **diversification** (Ruin-and-Recreate)
* Fast **evaluation via GPU**
* Effective **route reduction**
* Controlled **stochastic acceptance**

---



