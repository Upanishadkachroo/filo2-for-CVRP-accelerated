# instance/

## Overview

This module is responsible for:

- Loading and parsing CVRP instance data
- Providing access to problem attributes (coordinates, demands, capacity)
- Computing distances between vertices
- Precomputing nearest neighbors for sparsified search

It acts as the **data backbone** of the entire FILO2 solver.

---

## Files

| File | Description |
|------|------------|
| `Instance.hpp` | CPU-side representation of the CVRP instance |
| `Instance.cu`  | GPU-accelerated distance computation |
| `README.md`    | This documentation |

---

# CPU Design (Original FILO2)

## Responsibilities

The CPU version of `Instance`:

- Stores:
  - `xcoords`, `ycoords` → vertex positions
  - `demands` → customer demands
  - `neighbors` → k-nearest neighbor lists
- Provides:
  - `get_cost(i, j)` → Euclidean distance
  - `get_neighbors_of(i)` → candidate list

---

## Distance Computation

```cpp
inline double get_cost(int i, int j) const {
    const double dx = xcoords[i] - xcoords[j];
    const double dy = ycoords[i] - ycoords[j];
    const double dist = std::sqrt(dx*dx + dy*dy);
    return fastround(dist);  // (int)(dist + 0.5)
}