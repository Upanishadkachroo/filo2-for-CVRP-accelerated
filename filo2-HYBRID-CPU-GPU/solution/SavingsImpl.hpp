#ifndef COBRA_SAVINGS_IMPL_HPP
#define COBRA_SAVINGS_IMPL_HPP

#include <vector>
#include <algorithm>
#include "../instance/Instance.hpp"

#ifdef _OPENMP
#  include <omp.h>
#endif

namespace cobra {
namespace savings_impl {

struct Saving {
    int    i;
    int    j;
    double value;

    struct DescendingCmp {
        bool operator()(const Saving &a, const Saving &b) const noexcept {
            return a.value > b.value;
        }
    };
};

inline void compute_savings_parallel(
        const cobra::Instance &instance,
        double                 lambda,
        int                    neighbors_num,
        std::vector<Saving>   &out_savings)
{
    const int cbegin = instance.get_customers_begin();
    const int cend   = instance.get_customers_end();
    const int depot  = instance.get_depot();

#ifdef _OPENMP
    const int nthreads = omp_get_max_threads();
    std::vector<std::vector<Saving>> thread_savings(nthreads);
    const std::size_t expected =
        static_cast<std::size_t>(instance.get_customers_num()) *
        static_cast<std::size_t>(neighbors_num) / 2 / nthreads + 64;
    for (auto &ts : thread_savings) ts.reserve(expected);

    #pragma omp parallel for schedule(dynamic, 256) default(none) \
        shared(instance, thread_savings, lambda, neighbors_num, cbegin, cend, depot)
    for (int i = cbegin; i < cend; ++i) {
        auto &local = thread_savings[omp_get_thread_num()];
        const auto &nbrs = instance.get_neighbors_of(i);
        const double cost_i_depot = instance.get_cost(i, depot);
        for (unsigned n = 1, added = 0;
             added < static_cast<unsigned>(neighbors_num) && n < nbrs.size(); ++n) {
            const int j = nbrs[n];
            if (i < j) {
                local.push_back({i, j,
                    cost_i_depot + instance.get_cost(depot, j)
                    - lambda * instance.get_cost(i, j)});
                ++added;
            }
        }
    }
    for (auto &ts : thread_savings)
        out_savings.insert(out_savings.end(),
                           std::make_move_iterator(ts.begin()),
                           std::make_move_iterator(ts.end()));
#else
    for (int i = cbegin; i < cend; ++i) {
        const auto &nbrs = instance.get_neighbors_of(i);
        const double cost_i_depot = instance.get_cost(i, depot);
        for (unsigned n = 1, added = 0;
             added < static_cast<unsigned>(neighbors_num) && n < nbrs.size(); ++n) {
            const int j = nbrs[n];
            if (i < j) {
                out_savings.push_back({i, j,
                    cost_i_depot + instance.get_cost(depot, j)
                    - lambda * instance.get_cost(i, j)});
                ++added;
            }
        }
    }
#endif
}

inline void sort_savings(std::vector<Saving> &savings) {
    std::sort(savings.begin(), savings.end(), Saving::DescendingCmp{});
}

} // namespace savings_impl
} // namespace cobra

#endif
