#ifndef _FILO2_ROUTEMIN_HPP_
#define _FILO2_ROUTEMIN_HPP_

#include <chrono>
#include <limits>
#include <algorithm>

#include "../base/PrettyPrinter.hpp"
#include "../base/SparseIntSet.hpp"
#include "../instance/Instance.hpp"
#include "../localsearch/LocalSearch.hpp"
#include "../movegen/MoveGenerators.hpp"
#include "../solution/Solution.hpp"

inline cobra::Solution routemin(
    const cobra::Instance &instance,
    const cobra::Solution &source,
    std::mt19937 &rand_engine,
    cobra::MoveGenerators &move_generators,
    int kmin,
    int max_iter,
    double tolerance)
{
#ifdef VERBOSE
    auto t_start = std::chrono::high_resolution_clock::now();
#endif

    // Local search setup
    auto rvnd0 = cobra::RandomizedVariableNeighborhoodDescent<true>(
        instance, move_generators,
        {cobra::E11, cobra::E10, cobra::TAILS, cobra::SPLIT,
         cobra::RE22B, cobra::E22, cobra::RE20, cobra::RE21,
         cobra::RE22S, cobra::E21, cobra::E20, cobra::TWOPT,
         cobra::RE30, cobra::E30, cobra::RE33B, cobra::E33,
         cobra::RE31, cobra::RE32B, cobra::RE33S,
         cobra::E31, cobra::E32, cobra::RE32S},
        rand_engine, tolerance);

    auto local_search = cobra::VariableNeighborhoodDescentComposer(tolerance);
    local_search.append(&rvnd0);

    // Activate all move generators
    std::vector<int> gamma_vertices;
    std::vector<double> gamma(instance.get_vertices_num(), 1.0);

    for (int i = instance.get_vertices_begin(); i < instance.get_vertices_end(); i++) {
        gamma_vertices.push_back(i);
    }
    move_generators.set_active_percentage(gamma, gamma_vertices);

    cobra::Solution best_solution = source;
    cobra::Solution solution = source;

    std::uniform_real_distribution<double> uniform01(0.0, 1.0);
    std::uniform_int_distribution<int> cust_dist(
        instance.get_customers_begin(),
        instance.get_customers_end() - 1);

    // Cooling parameter
    double t = 1.0;
    double t_end = 0.01;
    double cooling = std::pow(t_end / t, 1.0 / max_iter);

    std::vector<int> removed;
    std::vector<int> still_removed;
    removed.reserve(instance.get_customers_num());
    still_removed.reserve(instance.get_customers_num());

    cobra::SparseIntSet neighbor_routes(instance.get_vertices_num());

    for (int iter = 0; iter < max_iter; iter++) {

        solution.clear_svc();

        // Select seed route
        int seed;
        do {
            seed = cust_dist(rand_engine);
        } while (!solution.is_customer_in_solution(seed));

        std::vector<int> selected_routes;
        selected_routes.push_back(solution.get_route_index(seed));

        const auto &neighbors = instance.get_neighbors_of(seed);

        for (size_t n = 1; n < neighbors.size(); n++) {
            int v = neighbors[n];
            if (v == instance.get_depot()) continue;
            if (!solution.is_customer_in_solution(v)) continue;

            int r = solution.get_route_index(v);
            if (r != selected_routes[0]) {
                selected_routes.push_back(r);
                break;
            }
        }

        // REMOVE phase
        removed.clear();
        removed.insert(removed.end(), still_removed.begin(), still_removed.end());
        still_removed.clear();

        for (int route : selected_routes) {

            int curr = solution.get_first_customer(route);

            while (curr != instance.get_depot()) {
                int next = solution.get_next_vertex(curr);
                solution.remove_vertex(route, curr);
                removed.push_back(curr);
                curr = next;
            }

            if (!solution.is_route_empty(route)) {
                solution.remove_route(route);
            }
        }

        // Shuffle removed customers
        if (rand_engine() % 2 == 0) {
            std::sort(removed.begin(), removed.end(),
                [&instance](int a, int b) {
                    return instance.get_demand(a) > instance.get_demand(b);
                });
        } else {
            std::shuffle(removed.begin(), removed.end(), rand_engine);
        }
        // REINSERT phase
        for (int cust : removed) {

            int best_route = -1;
            int best_where = -1;
            double best_delta = std::numeric_limits<double>::max();

            const auto &neigh = instance.get_neighbors_of(cust);

            neighbor_routes.clear();

            for (size_t n = 1; n < neigh.size(); n++) {
                int v = neigh[n];
                if (v == instance.get_depot()) continue;
                if (!solution.is_customer_in_solution(v)) continue;

                neighbor_routes.insert(solution.get_route_index(v));
            }

            double c_depot = instance.get_cost(cust, instance.get_depot());

            for (int route : neighbor_routes.get_elements()) {

                if (solution.get_route_load(route) + instance.get_demand(cust) >
                    instance.get_vehicle_capacity()) {
                    continue;
                }

                for (int j = solution.get_first_customer(route);
                     j != instance.get_depot();
                     j = solution.get_next_vertex(j)) {

                    int prev = solution.get_prev_vertex(route, j);

                    double delta =
                        -solution.get_cost_prev_customer(j)
                        + instance.get_cost(prev, cust)
                        + instance.get_cost(cust, j);

                    if (delta < best_delta) {
                        best_delta = delta;
                        best_route = route;
                        best_where = j;
                    }
                }

                double end_delta =
                    -solution.get_cost_prev_depot(route)
                    + instance.get_cost(solution.get_last_customer(route), cust)
                    + c_depot;

                if (end_delta < best_delta) {
                    best_delta = end_delta;
                    best_route = route;
                    best_where = instance.get_depot();
                }
            }

            // Decision: insert or delay
            if (best_route == -1) {

                double r = uniform01(rand_engine);

                if (r > t || solution.get_routes_num() < kmin) {
                    solution.build_one_customer_route(cust);
                } else {
                    still_removed.push_back(cust);
                }

            } else {
                solution.insert_vertex_before(best_route, best_where, cust);
            }
        }

        // LOCAL SEARCH
        local_search.sequential_apply(solution);


        // ACCEPT / UPDATE BEST
        if (still_removed.empty()) {

            if (solution.get_cost() < best_solution.get_cost() ||
                (solution.get_cost() == best_solution.get_cost() &&
                 solution.get_routes_num() < best_solution.get_routes_num())) {

                best_solution = solution;

                if (best_solution.get_routes_num() <= kmin) {
                    break;  // early exit
                }
            }
        }

        // RESET IF WORSE
        if (solution.get_cost() > best_solution.get_cost()) {
            solution = best_solution;
            still_removed.clear();
        }

        // Cooling
        t *= cooling;

        assert(solution.is_feasible());
    }

    assert(best_solution.is_feasible());
    return best_solution;
}

#endif