#include "Instance.hpp"
#include <algorithm>
#include "../base/KDTree.hpp"
#include "../base/Timer.hpp"

#ifdef VERBOSE
#include <iostream>
#endif

#ifdef _OPENMP
#include <omp.h>
#endif

#ifdef USE_CUDA_NEIGHBORS
    #include "../cuda/CudaNeighborFinder.hpp"
    #ifdef USE_GRID_NEIGHBOR
        #include "../cuda/GridNeighborFinder.hpp"
    #endif
#endif

namespace cobra {

std::optional<Instance> Instance::make(const std::string& filepath, int neighbors_num) {
    Parser parser(filepath);
    std::optional<Parser::Data> maybe_data = parser.Parse();
    if (!maybe_data.has_value()) {
        return std::nullopt;
    }
    return Instance(maybe_data.value(), neighbors_num);
}

Instance::Instance(const Parser::Data& data, int neighbors_num) {
    neighbors_num = std::min(neighbors_num, static_cast<int>(data.demands.size()));

    vehicle_capacity = data.vehicle_capacity;
    xcoords          = std::move(data.xcoords);
    ycoords          = std::move(data.ycoords);
    demands          = std::move(data.demands);

    neighbors.resize(get_vertices_num());

    const int N = get_vertices_num();

    // ----------------------------------------------------------------
    // Helper: post-process neighbors built by KDTree to ensure vertex
    // i is always in position 0 of neighbors[i].
    // ----------------------------------------------------------------
    auto fix_self = [&](int i) {
        if (neighbors[i].empty()) return;
        if (neighbors[i][0] != i) {
            int pos = 1;
            while (pos < static_cast<int>(neighbors[i].size())) {
                if (neighbors[i][pos] == i) break;
                ++pos;
            }
            std::swap(neighbors[i][0], neighbors[i][pos]);
        }
    };

    // ----------------------------------------------------------------
    // Helper: KDTree + OpenMP fallback (used for large N or CPU-only)
    // ----------------------------------------------------------------
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
                if (timer.elapsed_time<std::chrono::milliseconds>() > 10000 && progress != last_progress) {
                    std::cout << "Progress: " << progress << "%\n";
                    timer.reset();
                    last_progress = progress;
                }
            }
#endif
        }
    };

#ifdef USE_CUDA_NEIGHBORS
    // ----------------------------------------------------------------
    // Hybrid dispatcher:
    //   N <= 200,000              -> CUDA brute-force
    //   200,000 < N <= 1,000,000 -> CUDA grid-based  (requires -DUSE_GRID_NEIGHBOR)
    //   N > 1,000,000            -> KDTree + OpenMP
    // ----------------------------------------------------------------

    if (N <= 200000) {
#ifdef VERBOSE
        std::cout << "[Neighbor] Using CUDA brute-force (N=" << N << " <= 200000)\n";
#endif
        cobra::CudaNeighborFinder gpu_finder(xcoords, ycoords);
        auto flat = gpu_finder.computeAllNeighborsFlat(neighbors_num, true);

        if (flat.empty()) {
#ifdef VERBOSE
            std::cout << "[Neighbor] CUDA brute-force failed, falling back to KDTree+OpenMP\n";
#endif
            run_kdtree();
        } else {
            for (int i = 0; i < N; ++i) {
                neighbors[i].assign(flat.begin() + i * neighbors_num,
                                    flat.begin() + i * neighbors_num + neighbors_num);
                fix_self(i);
            }
        }

    }
#ifdef USE_GRID_NEIGHBOR
    else if (N <= 1000000) {
#ifdef VERBOSE
        std::cout << "[Neighbor] Using CUDA grid-based k-NN (200000 < N=" << N << " <= 1000000)\n";
#endif
        cobra::GridNeighborFinder grid_finder(xcoords, ycoords);
        auto flat = grid_finder.computeAllNeighborsFlat(neighbors_num, true);

        if (flat.empty()) {
#ifdef VERBOSE
            std::cout << "[Neighbor] CUDA grid k-NN failed, falling back to KDTree+OpenMP\n";
#endif
            run_kdtree();
        } else {
            for (int i = 0; i < N; ++i) {
                neighbors[i].assign(flat.begin() + i * neighbors_num,
                                    flat.begin() + i * neighbors_num + neighbors_num);
                fix_self(i);
            }
        }
    }
#endif // USE_GRID_NEIGHBOR
    else {
#ifdef VERBOSE
        std::cout << "[Neighbor] N=" << N << " > 1000000, using KDTree+OpenMP fallback\n";
#endif
        run_kdtree();
    }

#else
    // ----------------------------------------------------------------
    // CUDA disabled: always use KDTree + OpenMP
    // ----------------------------------------------------------------
    run_kdtree();
#endif // USE_CUDA_NEIGHBORS
}

} // namespace cobra