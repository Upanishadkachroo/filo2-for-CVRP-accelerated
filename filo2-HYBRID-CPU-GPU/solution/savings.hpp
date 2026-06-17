#ifndef _FILO2_SOLUTIONALGORITHMS_HPP_
#define _FILO2_SOLUTIONALGORITHMS_HPP_

#include "../base/Timer.hpp"
#include "Solution.hpp"
#include "../instance/Saving.hpp"

#ifdef _OPENMP
#include <omp.h>
#endif

#ifdef USE_CUDA_NEIGHBORS
#include "../cuda/SavingsKernel.cuh"
#endif

#ifdef __GNUC__
#include <parallel/algorithm>
#endif

namespace cobra {

    inline void clarke_and_wright(const Instance &instance, Solution &solution, const double lambda, int neighbors_num) {

        solution.reset();

        for (auto i = instance.get_customers_begin(); i < instance.get_customers_end(); i++) {
            solution.build_one_customer_route</*record_acion=*/false>(i);
        }
        assert(solution.is_feasible());

        neighbors_num = std::min(instance.get_customers_num() - 1, neighbors_num);

        std::vector<Saving> savings;

        const int N = instance.get_customers_num();

#ifdef USE_CUDA_NEIGHBORS
        // Use GPU for large instances (threshold empirically set)
        if (N > 200000) {
            if (computeSavingsGPU(instance, neighbors_num, lambda, savings)) {
                // savings is sorted descending; skip CPU compute + sort
                goto merge_loop;
            }
            // If GPU fails, fall back to CPU (OpenMP or serial)
        }
#endif

        // ---------- CPU savings computation (OpenMP if available) ----------
        {
#ifdef _OPENMP
            int num_threads = omp_get_max_threads();
            std::vector<std::vector<Saving>> thread_savings(num_threads);

            #pragma omp parallel
            {
                int tid = omp_get_thread_num();
                #pragma omp for schedule(dynamic)
                for (auto i = instance.get_customers_begin(); i < instance.get_customers_end(); i++) {
                    for (auto n = 1u, added = 0u; added < static_cast<unsigned int>(neighbors_num) && n < instance.get_neighbors_of(i).size(); n++) {
                        const auto j = instance.get_neighbors_of(i)[n];
                        if (i < j) {
                            const double value = instance.get_cost(i, instance.get_depot()) +
                                                 instance.get_cost(instance.get_depot(), j) -
                                                 lambda * instance.get_cost(i, j);
                            thread_savings[tid].push_back({i, j, value});
                            added++;
                        }
                    }
                }
            }

            // Merge thread‑local savings into global vector
            size_t total = 0;
            for (int t = 0; t < num_threads; ++t) total += thread_savings[t].size();
            savings.reserve(total);
            for (int t = 0; t < num_threads; ++t) {
                savings.insert(savings.end(), thread_savings[t].begin(), thread_savings[t].end());
            }
#else
            // Serial fallback
            savings.reserve(instance.get_customers_num() * neighbors_num);
            for (auto i = instance.get_customers_begin(); i < instance.get_customers_end(); i++) {
                for (auto n = 1u, added = 0u; added < static_cast<unsigned int>(neighbors_num) && n < instance.get_neighbors_of(i).size(); n++) {
                    const auto j = instance.get_neighbors_of(i)[n];
                    if (i < j) {
                        const double value = instance.get_cost(i, instance.get_depot()) +
                                             instance.get_cost(instance.get_depot(), j) -
                                             lambda * instance.get_cost(i, j);
                        savings.push_back({i, j, value});
                        added++;
                    }
                }
            }
#endif

// ---------- Corrected preprocessor directive ----------
#if defined(_OPENMP) && defined(__GNUC__)
            __gnu_parallel::sort(savings.begin(), savings.end(),
                                 [](const Saving &a, const Saving &b) { return a.value > b.value; });
#else
            std::sort(savings.begin(), savings.end(), [](const Saving &a, const Saving &b) { return a.value > b.value; });
#endif
        }

#ifdef USE_CUDA_NEIGHBORS
merge_loop:
#endif
        // ---------- Phase 3: Sequential route merging ----------
#ifdef VERBOSE
        Timer timer;
#endif
        for (auto n = 0; n < static_cast<int>(savings.size()); ++n) {
            const auto &saving = savings[n];
            const auto i = saving.i;
            const auto j = saving.j;
            const auto iRoute = solution.get_route_index(i);
            const auto jRoute = solution.get_route_index(j);
            if (iRoute == jRoute) continue;
            if (solution.get_last_customer(iRoute) == i && solution.get_first_customer(jRoute) == j &&
                solution.get_route_load(iRoute) + solution.get_route_load(jRoute) <= instance.get_vehicle_capacity()) {
                solution.append_route(iRoute, jRoute);
            } else if (solution.get_last_customer(jRoute) == j && solution.get_first_customer(iRoute) == i &&
                       solution.get_route_load(iRoute) + solution.get_route_load(jRoute) <= instance.get_vehicle_capacity()) {
                solution.append_route(jRoute, iRoute);
            }
#ifdef VERBOSE
            if (timer.elapsed_time<std::chrono::seconds>() > 2) {
                std::cout << "Progress: " << 100.0 * (n + 1) / savings.size() << "%, Solution cost: " << solution.get_cost() << " \n";
                timer.reset();
            }
#endif
        }
        assert(solution.is_feasible());
    }

}  // namespace cobra

#endif
