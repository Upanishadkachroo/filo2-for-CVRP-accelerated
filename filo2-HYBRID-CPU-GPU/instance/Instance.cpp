#include "Instance.hpp"

#include <algorithm>

#include "../base/KDTree.hpp"
#include "../base/Timer.hpp"

// CUDA headers
#ifdef USE_CUDA_NEIGHBORS
    #include "../cuda/CudaNeighborFinder.hpp"   // brute‑force implementation
    #include "cuda/uniform_grid.cuh"            // new uniform‑grid implementation
#endif

#ifdef VERBOSE
    #include <iostream>
#endif

#ifdef _OPENMP
    #include <omp.h>
#endif

namespace cobra {

// -------------------------------------------------------------------
// Static factory method
// -------------------------------------------------------------------
std::optional<Instance> Instance::make(const std::string& filepath, int neighbors_num) {
    Parser parser(filepath);
    std::optional<Parser::Data> maybe_data = parser.Parse();
    if (!maybe_data.has_value()) {
        return std::nullopt;
    }
    return Instance(maybe_data.value(), neighbors_num);
}

// -------------------------------------------------------------------
// Constructor – dispatches to the best available neighbor search
// -------------------------------------------------------------------
Instance::Instance(const Parser::Data& data, int neighbors_num) {
    // Clamp k to the number of vertices
    neighbors_num = std::min(neighbors_num, static_cast<int>(data.demands.size()));

    // Copy basic data
    vehicle_capacity = data.vehicle_capacity;
    xcoords          = std::move(data.xcoords);
    ycoords          = std::move(data.ycoords);
    demands          = std::move(data.demands);

    const int N = get_vertices_num();
    neighbors.resize(N);

    // -----------------------------------------------------------------
    // Helper: ensure vertex i is at position 0 of neighbors[i]
    // -----------------------------------------------------------------
    auto fix_self = [&](int i) {
        if (neighbors[i].empty()) return;
        if (neighbors[i][0] != i) {
            int pos = 1;
            while (pos < static_cast<int>(neighbors[i].size())) {
                if (neighbors[i][pos] == i) break;
                ++pos;
            }
            // If not found (shouldn't happen), we keep the first element
            if (pos < static_cast<int>(neighbors[i].size())) {
                std::swap(neighbors[i][0], neighbors[i][pos]);
            }
        }
    };

    // -----------------------------------------------------------------
    // Fallback: KDTree + OpenMP (used for large N or when CUDA fails)
    // -----------------------------------------------------------------
    auto run_kdtree = [&]() {
        KDTree kd_tree(xcoords, ycoords);

#ifdef VERBOSE
        Timer timer;
        int last_progress = -1;
#endif

        #pragma omp parallel for schedule(dynamic)
        for (int i = get_vertices_begin(); i < get_vertices_end(); ++i) {
            neighbors[i] = kd_tree.GetNearestNeighbors(xcoords[i], ycoords[i], neighbors_num);
            fix_self(i);

#ifdef VERBOSE
            #pragma omp critical
            {
                int progress = 100 * (i + 1) / N;
                if (timer.elapsed_time<std::chrono::milliseconds>() > 10000 &&
                    progress != last_progress) {
                    std::cout << "[KDTree] Progress: " << progress << "%\n";
                    timer.reset();
                    last_progress = progress;
                }
            }
#endif
        }
    };

    // -----------------------------------------------------------------
    // Hybrid dispatcher based on N and available CUDA implementations
    // -----------------------------------------------------------------
#ifdef USE_CUDA_NEIGHBORS
    // Case 1: Small N → brute‑force CUDA (O(N²) but fast for tiny N)
    if (N <= 200000) {
#ifdef VERBOSE
        std::cout << "[Neighbor] Using CUDA brute‑force (N = " << N << " ≤ 200000)\n";
#endif
        cobra::CudaNeighborFinder gpu_finder(xcoords, ycoords);
        auto flat = gpu_finder.computeAllNeighborsFlat(neighbors_num, /* includeSelf = */ true);

        if (flat.empty()) {
#ifdef VERBOSE
            std::cout << "[Neighbor] CUDA brute‑force failed. Falling back to KDTree.\n";
#endif
            run_kdtree();
        } else {
            // Copy flat result into vector-of-vectors
            for (int i = 0; i < N; ++i) {
                const int offset = i * neighbors_num;
                neighbors[i].assign(flat.begin() + offset,
                                    flat.begin() + offset + neighbors_num);
                fix_self(i);
            }
        }
    }
    // Case 2: Medium N → uniform‑grid (exact, GPU‑friendly)
    else if (N <= 1000000) {
#ifdef VERBOSE
        std::cout << "[Neighbor] Using CUDA uniform grid (200000 < N = " << N << " ≤ 1000000)\n";
#endif
        UniformGridNeighbors grid;
        if (!grid.build(xcoords, ycoords)) {
#ifdef VERBOSE
            std::cout << "[Neighbor] Grid build failed. Falling back to KDTree.\n";
#endif
            run_kdtree();
        } else {
            std::vector<int> flat;
            if (!grid.computeNeighbors(neighbors_num, flat)) {
#ifdef VERBOSE
                std::cout << "[Neighbor] Grid query failed. Falling back to KDTree.\n";
#endif
                run_kdtree();
            } else {
                for (int i = 0; i < N; ++i) {
                    const int offset = i * neighbors_num;
                    neighbors[i].assign(flat.begin() + offset,
                                        flat.begin() + offset + neighbors_num);
                    fix_self(i);
                }
            }
        }
    }
    // Case 3: Very large N → stick with KDTree (avoids GPU memory issues)
    else {
#ifdef VERBOSE
        std::cout << "[Neighbor] N = " << N << " > 1000000. Using KDTree + OpenMP.\n";
#endif
        run_kdtree();
    }

#else // USE_CUDA_NEIGHBORS not defined → CPU only
    run_kdtree();
#endif
}

} // namespace cobra