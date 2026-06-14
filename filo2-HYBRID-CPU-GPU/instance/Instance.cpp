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
    #ifdef USE_GRID_NEIGHBOR
        #include "../cuda/GridNeighborFinder.hpp"
    #else
        #include "../cuda/CudaNeighborFinder.hpp"
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

        // Move data from parser (O(1), not parallelized)
        vehicle_capacity = data.vehicle_capacity;
        xcoords = std::move(data.xcoords);
        ycoords = std::move(data.ycoords);
        demands = std::move(data.demands);

        neighbors.resize(get_vertices_num());

#ifdef USE_CUDA_NEIGHBORS
        const int N = get_vertices_num();
        // Choose method based on instance size (thresholds are tunable)
        const int BRUTE_FORCE_MAX = 200000;   // use brute‑force CUDA for N <= 200k
        const int GRID_MAX = 1000000;         // use grid CUDA for N <= 1M

        if (N <= BRUTE_FORCE_MAX) {
            // ---------- Brute‑force batched CUDA (good for small/medium instances) ----------
            #ifdef USE_GRID_NEIGHBOR
                // Even if grid is available, brute‑force may be faster for small N
                cobra::CudaNeighborFinder gpu_finder(xcoords, ycoords);
                neighbors = gpu_finder.computeAllNeighbors(neighbors_num, true);
            #else
                cobra::CudaNeighborFinder gpu_finder(xcoords, ycoords);
                neighbors = gpu_finder.computeAllNeighbors(neighbors_num, true);
            #endif
        } 
        #ifdef USE_GRID_NEIGHBOR
        else if (N <= GRID_MAX) {
            // ---------- Grid‑based CUDA (scales to large instances) ----------
            cobra::GridNeighborFinder gpu_finder(xcoords, ycoords);
            neighbors = gpu_finder.computeAllNeighbors(neighbors_num, true);
        }
        #endif
        else {
            // ---------- Fallback to KDTree + OpenMP (CPU, no GPU memory issues) ----------
            KDTree kd_tree(xcoords, ycoords);
#ifdef VERBOSE
            Timer timer;
            int last_progress = -1;
#endif
            #pragma omp parallel for schedule(dynamic)
            for (int i = get_vertices_begin(); i < get_vertices_end(); ++i) {
                neighbors[i] = kd_tree.GetNearestNeighbors(xcoords[i], ycoords[i], neighbors_num);
                if (neighbors[i][0] != i) {
                    int n = 1;
                    while (n < static_cast<int>(neighbors[i].size())) {
                        if (neighbors[i][n] == i) break;
                        ++n;
                    }
                    std::swap(neighbors[i][0], neighbors[i][n]);
                }
#ifdef VERBOSE
                #pragma omp critical
                {
                    int progress = 100 * (i + 1) / get_vertices_num();
                    if (timer.elapsed_time<std::chrono::milliseconds>() > 10000 && progress != last_progress) {
                        std::cout << "Progress: " << progress << "%\n";
                        timer.reset();
                        last_progress = progress;
                    }
                }
#endif
            }
        }
#else
        // ---------- Original KDTree + OpenMP (no CUDA) ----------
        KDTree kd_tree(xcoords, ycoords);
#ifdef VERBOSE
        Timer timer;
        int last_progress = -1;
#endif
        #pragma omp parallel for schedule(dynamic)
        for (int i = get_vertices_begin(); i < get_vertices_end(); ++i) {
            neighbors[i] = kd_tree.GetNearestNeighbors(xcoords[i], ycoords[i], neighbors_num);
            if (neighbors[i][0] != i) {
                int n = 1;
                while (n < static_cast<int>(neighbors[i].size())) {
                    if (neighbors[i][n] == i) break;
                    ++n;
                }
                std::swap(neighbors[i][0], neighbors[i][n]);
            }
#ifdef VERBOSE
            #pragma omp critical
            {
                int progress = 100 * (i + 1) / get_vertices_num();
                if (timer.elapsed_time<std::chrono::milliseconds>() > 10000 && progress != last_progress) {
                    std::cout << "Progress: " << progress << "%\n";
                    timer.reset();
                    last_progress = progress;
                }
            }
#endif
        }
#endif // USE_CUDA_NEIGHBORS
    }

} // namespace cobra