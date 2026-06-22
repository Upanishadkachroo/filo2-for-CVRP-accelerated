#ifndef _FILO2_BPP_HPP_
#define _FILO2_BPP_HPP_

#include <algorithm>
#include <climits>
#include <set>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#include <parallel/algorithm>   // for __gnu_parallel::sort
#endif

#include "../instance/Instance.hpp"

namespace bpp {

    // ----------------------------------------------------------------
    // greedy_first_fit_decreasing — O(N log N) replacement
    // ----------------------------------------------------------------
    inline int greedy_first_fit_decreasing(const cobra::Instance& instance) {

        const int capacity = instance.get_vehicle_capacity();
        const int N        = instance.get_customers_num();

        // --- Step 1: build customer index array ---
        std::vector<int> customers(N);
        for (int i = 0; i < N; ++i)
            customers[i] = instance.get_customers_begin() + i;

        // --- Step 2: sort by demand descending (parallel if available) ---
        auto cmp = [&instance](int a, int b) {
            return instance.get_demand(a) > instance.get_demand(b);
        };

#if defined(_OPENMP) && defined(__GNUC__)
        __gnu_parallel::sort(customers.begin(), customers.end(), cmp);
#else
        std::sort(customers.begin(), customers.end(), cmp);
#endif

        // --- Step 3: FFD with O(log N) bin lookup ---
        std::multiset<std::pair<int, int>> open_bins;
        int used_bins = 0;

        for (int cust : customers) {
            const int demand = instance.get_demand(cust);

            // Find smallest remaining_capacity >= demand
            auto it = open_bins.lower_bound({demand, INT_MIN});

            if (it == open_bins.end()) {
                // No fitting bin exists — open a new one
                const int remaining = capacity - demand;
                if (remaining > 0)
                    open_bins.insert({remaining, used_bins});
                // else: bin is exactly full; no need to store 0 capacity
                ++used_bins;
            } else {
                // Reuse existing bin with updated remaining capacity
                const int new_remaining = it->first - demand;
                const int bin_id        = it->second;
                open_bins.erase(it);
                if (new_remaining > 0)
                    open_bins.insert({new_remaining, bin_id});
                // else: bin becomes full; we can discard it
            }
        }

        return used_bins;
    }

    // ----------------------------------------------------------------
    // lower_bound_routes — O(N) absolute lower bound
    // ----------------------------------------------------------------
    inline int lower_bound_routes(const cobra::Instance& instance) {
        long long total = 0;

#ifdef _OPENMP
        #pragma omp parallel for reduction(+:total) schedule(static)
#endif
        for (int i = instance.get_customers_begin();
                 i < instance.get_customers_end(); ++i) {
            total += instance.get_demand(i);
        }

        const int capacity = instance.get_vehicle_capacity();
        return static_cast<int>((total + capacity - 1) / capacity);
    }

}  // namespace bpp

#endif
