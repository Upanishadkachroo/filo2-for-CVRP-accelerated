/**
 * savings.hpp  (filo2 — Phase 1 parallelized)
 *
 * Public API is IDENTICAL to the original: same function signature, same
 * assert(solution.is_feasible()) postcondition, same VERBOSE progress output.
 *
 * What changed
 * ------------
 * The savings-computation loop and sort that were duplicated verbatim from
 * cobra/src/solution/SolutionConstruction.cpp have been replaced by calls to
 * cobra::savings_impl helpers (SavingsImpl.hpp).  This satisfies DRY — there
 * is now exactly ONE copy of the parallel savings logic.
 *
 * Parallelization summary for Phase 1 (this file's responsibility)
 * ----------------------------------------------------------------
 *   Step 1 — build_one_customer_route  : sequential (mutates Solution)
 *   Step 2 — compute_savings_parallel  : OpenMP parallel-for per customer
 *   Step 3 — sort_savings              : Thrust GPU sort (with fallback)
 *   Step 4 — greedy merge loop         : sequential (Solution not thread-safe)
 *
 * Build flags (forwarded from cobra CMake targets)
 *   -fopenmp            → OpenMP savings computation
 *   -DCOBRA_USE_THRUST  → Thrust GPU sort
 */

#ifndef _FILO2_SOLUTIONALGORITHMS_HPP_
#define _FILO2_SOLUTIONALGORITHMS_HPP_

#include "../base/Timer.hpp"
#include "Solution.hpp"
#include <cobra/SavingsImpl.hpp>   // shared Saving + parallel helpers (DRY)

namespace cobra {

    /**
     * Limited Clarke & Wright savings algorithm.
     *
     * Parallelized: savings computation uses OpenMP; sort uses Thrust when
     * COBRA_USE_THRUST is defined.  The greedy merge loop remains sequential.
     */
    inline void clarke_and_wright(
            const Instance &instance,
            Solution       &solution,
            const double    lambda,
            int             neighbors_num)
    {
        // ── Step 1: one-customer routes ───────────────────────────────────────
        solution.reset();

        for (auto i = instance.get_customers_begin(); i < instance.get_customers_end(); ++i) {
            solution.build_one_customer_route</*record_action=*/false>(i);
        }
        assert(solution.is_feasible());

        neighbors_num = std::min(instance.get_customers_num() - 1, neighbors_num);

        // ── Step 2: parallel savings computation ──────────────────────────────
        const auto savings_upper_bound =
            static_cast<std::size_t>(instance.get_customers_num()) *
            static_cast<std::size_t>(neighbors_num) / 2 + 1;

        std::vector<savings_impl::Saving> savings;
        savings.reserve(savings_upper_bound);

        savings_impl::compute_savings_parallel(instance, lambda, neighbors_num, savings);

        // ── Step 3: sort descending (Thrust GPU or std::sort) ─────────────────
        savings_impl::sort_savings(savings);

        // ── Step 4: greedy merge loop (sequential) ────────────────────────────
#ifdef VERBOSE
        Timer timer;
#endif
        const int cap = instance.get_vehicle_capacity();

        for (auto n = 0; n < static_cast<int>(savings.size()); ++n) {

            const auto &saving = savings[n];
            const int i = saving.i;
            const int j = saving.j;

            const int iRoute = solution.get_route_index(i);
            const int jRoute = solution.get_route_index(j);

            if (iRoute == jRoute) { continue; }

            if (solution.get_last_customer(iRoute)  == i &&
                solution.get_first_customer(jRoute) == j &&
                solution.get_route_load(iRoute) + solution.get_route_load(jRoute) <= cap)
            {
                solution.append_route(iRoute, jRoute);

            } else if (solution.get_last_customer(jRoute)  == j &&
                       solution.get_first_customer(iRoute) == i &&
                       solution.get_route_load(iRoute) + solution.get_route_load(jRoute) <= cap)
            {
                solution.append_route(jRoute, iRoute);
            }

#ifdef VERBOSE
            if (timer.elapsed_time<std::chrono::seconds>() > 2) {
                std::cout << "Progress: "
                          << 100.0 * (n + 1) / savings.size()
                          << "%, Solution cost: " << solution.get_cost() << " \n";
                timer.reset();
            }
#endif
        }

        assert(solution.is_feasible());
    }

}  // namespace cobra

#endif  // _FILO2_SOLUTIONALGORITHMS_HPP_