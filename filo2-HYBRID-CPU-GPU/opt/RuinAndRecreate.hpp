#ifndef _FILO2_RUINANDRECREATE_HPP_
#define _FILO2_RUINANDRECREATE_HPP_

#include <algorithm>
#include <random>
#include <limits>
#include <cassert>

#include "../base/SparseIntSet.hpp"
#include "../instance/Instance.hpp"
#include "../solution/Solution.hpp"

class RuinAndRecreate {

public:
    RuinAndRecreate(const cobra::Instance& instance_, std::mt19937& rand_engine_)
        : instance(instance_)
        , rand_engine(rand_engine_)
        , boolean_dist(0, 1)
        , customers_distribution(instance.get_customers_begin(), instance.get_customers_end() - 1)
        , rand_uniform(0, 3)
        , routes(instance.get_vertices_num()) { }

    int apply(cobra::Solution& solution, std::vector<int>& omega) {

        assert(solution.is_feasible());

        removed.clear();
        routes.clear();

        int seed = customers_distribution(rand_engine);
        int N = omega[seed];

        int curr = seed;

        // RUIN PHASE
        for (int n = 0; n < N; n++) {

            if (curr == instance.get_depot()) break;

            int route = solution.get_route_index(curr);
            int next = cobra::Solution::dummy_vertex;

            removed.push_back(curr);
            routes.insert(route);

            if (solution.get_route_size(route) > 1 && boolean_dist(rand_engine)) {

                if (boolean_dist(rand_engine)) {
                    next = solution.get_next_vertex(curr);
                    if (next == instance.get_depot()) {
                        next = solution.get_next_vertex(route, next);
                    }
                } else {
                    next = solution.get_prev_vertex(curr);
                    if (next == instance.get_depot()) {
                        next = solution.get_prev_vertex(route, next);
                    }
                }

            }
            else {

                const auto& neigh = instance.get_neighbors_of(curr);

                for (size_t m = 1; m < neigh.size(); m++) {

                    int neighbor = neigh[m];

                    if (neighbor == instance.get_depot()) continue;
                    if (!solution.is_customer_in_solution(neighbor)) continue;

                    int neigh_route = solution.get_route_index(neighbor);

                    if (!boolean_dist(rand_engine)) {
                        // Allow revisiting routes
                        next = neighbor;
                        break;
                    } else if (!routes.contains(neigh_route)) {
                        next = neighbor;
                        break;
                    }
                }
            }

            // Remove current
            solution.remove_vertex(route, curr);

            if (solution.is_route_empty(route)) {
                solution.remove_route(route);
            }

            if (next == cobra::Solution::dummy_vertex) break;

            curr = next;
        }

        // SHUFFLE STRATEGY
        switch (rand_uniform(rand_engine)) {
        case 0:
            std::shuffle(removed.begin(), removed.end(), rand_engine);
            break;

        case 1:
            std::sort(removed.begin(), removed.end(),
                [this](int a, int b) {
                    return instance.get_demand(a) > instance.get_demand(b);
                });
            break;

        case 2:
            std::sort(removed.begin(), removed.end(),
                [this](int a, int b) {
                    return instance.get_cost(a, instance.get_depot()) >
                           instance.get_cost(b, instance.get_depot());
                });
            break;

        case 3:
            std::sort(removed.begin(), removed.end(),
                [this](int a, int b) {
                    return instance.get_cost(a, instance.get_depot()) <
                           instance.get_cost(b, instance.get_depot());
                });
            break;
        }

        assert(solution.is_feasible());

        // RECREATE PHASE
        for (int customer : removed) {

            int best_route = cobra::Solution::dummy_route;
            int best_where = cobra::Solution::dummy_vertex;
            double best_cost = std::numeric_limits<double>::max();

            const auto& neighbors = instance.get_neighbors_of(customer);
            routes.clear();

            // Collect candidate routes
            for (size_t n = 1; n < neighbors.size(); n++) {
                int where = neighbors[n];

                if (where == instance.get_depot()) continue;
                if (!solution.is_customer_in_solution(where)) continue;

                routes.insert(solution.get_route_index(where));
            }

            double c_depot = instance.get_cost(customer, instance.get_depot());

            for (int route : routes.get_elements()) {

                if (solution.get_route_load(route) + instance.get_demand(customer) >
                    instance.get_vehicle_capacity()) {
                    continue;
                }

                // Try insertion positions
                for (int where = solution.get_first_customer(route);
                     where != instance.get_depot();
                     where = solution.get_next_vertex(where)) {

                    int prev = solution.get_prev_vertex(where);

                    double cost =
                        -solution.get_cost_prev_customer(where)
                        + instance.get_cost(prev, customer)
                        + instance.get_cost(customer, where);

                    if (cost < best_cost) {
                        best_cost = cost;
                        best_route = route;
                        best_where = where;
                    }
                }

                // Insert at end
                double end_cost =
                    -solution.get_cost_prev_depot(route)
                    + instance.get_cost(solution.get_last_customer(route), customer)
                    + c_depot;

                if (end_cost < best_cost) {
                    best_cost = end_cost;
                    best_route = route;
                    best_where = instance.get_depot();
                }
            }

            // Decision: new route vs insert
            if (best_route == cobra::Solution::dummy_route ||
                (2.0 * c_depot < best_cost)) {

                solution.build_one_customer_route(customer);

            } else {
                solution.insert_vertex_before(best_route, best_where, customer);
            }

            assert(solution.is_feasible());
        }

        return seed;
    }

private:
    const cobra::Instance& instance;
    std::mt19937& rand_engine;

    std::uniform_int_distribution<int> boolean_dist;
    std::uniform_int_distribution<int> customers_distribution;
    std::uniform_int_distribution<int> rand_uniform;

    std::vector<int> removed;
    cobra::SparseIntSet routes;
};

#endif