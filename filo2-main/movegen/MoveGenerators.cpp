#include "MoveGenerators.hpp"

#include <chrono>
#include <iomanip>
#include <iostream>

// Uncomment if using OpenMP
// #include <omp.h>

namespace cobra {

// ---------------------------------------------------------------------
// Constructor with profiling
// ---------------------------------------------------------------------
MoveGenerators::MoveGenerators(const Instance& instance_, int k)
    : max_num_neighbors(std::min(k, instance_.get_vertices_num() - 1))
    , heap(MoveGeneratorsHeap())
    , vertex_timestamp(instance_.get_vertices_num(), 0)
    , vertices_in_updated_moves(instance_.get_vertices_num())
    , unique_endpoints(instance_.get_vertices_num()) {

    auto total_start = std::chrono::high_resolution_clock::now();

    // -----------------------------------------------------------------
    // PASS 1: Generate candidate base moves
    // -----------------------------------------------------------------
    auto pass1_start = std::chrono::high_resolution_clock::now();

    const int neighbors_begin = 1;
    const int neighbors_end = neighbors_begin + max_num_neighbors;
    const int V = instance_.get_vertices_num();

    // Collect candidates as (i, j, cost) with i < j (using original tie‑breaking)
    std::vector<std::tuple<int, int, double>> candidates;
    candidates.reserve(static_cast<size_t>(V) * max_num_neighbors);

#ifdef _OPENMP
    #pragma omp parallel
    {
        std::vector<std::tuple<int, int, double>> local_candidates;
        local_candidates.reserve(static_cast<size_t>(V) * max_num_neighbors / omp_get_num_threads());

        #pragma omp for nowait
        for (int i = instance_.get_vertices_begin(); i < instance_.get_vertices_end(); ++i) {
            const auto& ineighbors = instance_.get_neighbors_of(i);
            for (int p = neighbors_begin; p < neighbors_end; ++p) {
                assert(p < static_cast<int>(ineighbors.size()));
                const int j = ineighbors[p];
                const double cost = instance_.get_cost(i, j);

                // Original logic: decide orientation to avoid duplicates
                if (i < j) {
                    local_candidates.emplace_back(i, j, cost);
                } else {
                    const auto& jneighbors = instance_.get_neighbors_of(j);
                    const double cij = cost;
                    const double cjn = instance_.get_cost(j, jneighbors[neighbors_end - 1]);
                    if (cij > cjn) {
                        local_candidates.emplace_back(j, i, cost);
                    } else if (std::fabs(cij - cjn) < 0.00001) {
                        // We'll dedup later
                        local_candidates.emplace_back(j, i, cost);
                    }
                }
            }
        }

        #pragma omp critical
        {
            candidates.insert(candidates.end(), local_candidates.begin(), local_candidates.end());
        }
    }
#else
    // Sequential version
    for (int i = instance_.get_vertices_begin(); i < instance_.get_vertices_end(); ++i) {
        const auto& ineighbors = instance_.get_neighbors_of(i);
        for (int p = neighbors_begin; p < neighbors_end; ++p) {
            assert(p < static_cast<int>(ineighbors.size()));
            const int j = ineighbors[p];
            const double cost = instance_.get_cost(i, j);

            if (i < j) {
                candidates.emplace_back(i, j, cost);
            } else {
                const auto& jneighbors = instance_.get_neighbors_of(j);
                const double cij = cost;
                const double cjn = instance_.get_cost(j, jneighbors[neighbors_end - 1]);
                if (cij > cjn) {
                    candidates.emplace_back(j, i, cost);
                } else if (std::fabs(cij - cjn) < 0.00001) {
                    candidates.emplace_back(j, i, cost);
                }
            }
        }
    }
#endif

    auto pass1_end = std::chrono::high_resolution_clock::now();
    profile_data.pass1_ms = std::chrono::duration_cast<std::chrono::milliseconds>(pass1_end - pass1_start).count();

    // -----------------------------------------------------------------
    // PASS 2: Sort and deduplicate
    // -----------------------------------------------------------------
    auto pass2_start = std::chrono::high_resolution_clock::now();

    std::sort(candidates.begin(), candidates.end(),
              [](const auto& a, const auto& b) {
                  if (std::get<0>(a) != std::get<0>(b)) return std::get<0>(a) < std::get<0>(b);
                  if (std::get<1>(a) != std::get<1>(b)) return std::get<1>(a) < std::get<1>(b);
                  return std::get<2>(a) < std::get<2>(b);
              });

    std::vector<std::tuple<int, int, double>> unique_candidates;
    unique_candidates.reserve(candidates.size());
    for (size_t idx = 0; idx < candidates.size(); ) {
        int i = std::get<0>(candidates[idx]);
        int j = std::get<1>(candidates[idx]);
        double cost = std::get<2>(candidates[idx]);
        unique_candidates.emplace_back(i, j, cost);
        while (idx < candidates.size() && std::get<0>(candidates[idx]) == i && std::get<1>(candidates[idx]) == j) {
            ++idx;
        }
    }

    moves.reserve(unique_candidates.size() * 2);
    edge_costs.reserve(unique_candidates.size());
    base_move_indices_involving.resize(V);

    for (const auto& [i, j, cost] : unique_candidates) {
        const int base_idx = moves.size();
        moves.emplace_back(i, j);
        moves.emplace_back(j, i);
        edge_costs.emplace_back(cost);
        base_move_indices_involving[i].push_back(base_idx);
        base_move_indices_involving[j].push_back(base_idx);
    }

    auto pass2_end = std::chrono::high_resolution_clock::now();
    profile_data.pass2_ms = std::chrono::duration_cast<std::chrono::milliseconds>(pass2_end - pass2_start).count();

    // -----------------------------------------------------------------
    // LOOP 2: Sort per vertex's base indices by cost
    // -----------------------------------------------------------------
    auto loop2_start = std::chrono::high_resolution_clock::now();

    for (int i = instance_.get_vertices_begin(); i < instance_.get_vertices_end(); ++i) {
        std::sort(base_move_indices_involving[i].begin(), base_move_indices_involving[i].end(),
                  [this](int a, int b) {
                      const auto& a_move = get(a);
                      const double a_cost = get_edge_cost(a_move);
                      const auto& b_move = get(b);
                      const double b_cost = get_edge_cost(b_move);
                      return a_cost < b_cost;
                  });
    }

    auto loop2_end = std::chrono::high_resolution_clock::now();
    profile_data.loop2_ms = std::chrono::duration_cast<std::chrono::milliseconds>(loop2_end - loop2_start).count();

    // -----------------------------------------------------------------
    // Finish initialization (as original)
    // -----------------------------------------------------------------
    update_bits.resize(V, 2);
    active_move_indices_involving_1st.resize(V);
    current_num_neighbors.resize(V, 0);
    move_active_in_1st.resize(moves.size() / 2, false);
    move_active_in_2nd.resize(moves.size() / 2, false);

    auto total_end = std::chrono::high_resolution_clock::now();
    profile_data.total_constructor_ms = std::chrono::duration_cast<std::chrono::milliseconds>(total_end - total_start).count();
    profile_data.total_moves_generated = moves.size();
    profile_data.unique_base_moves = moves.size() / 2;

    print_profile();
}

// ---------------------------------------------------------------------
// set_active_percentage implementation (moved from header)
// ---------------------------------------------------------------------
void MoveGenerators::set_active_percentage(std::vector<double>& percentage, std::vector<int>& vertices) {
    vertices_getting_updated.clear();
    vertices_in_updated_moves.clear();

    for (const int vertex : vertices) {
        // Convert the percentage to the number of neighbors to consider.
        const int num_neigbors = std::round(percentage[vertex] * max_num_neighbors);
        assert(num_neigbors <= static_cast<int>(base_move_indices_involving[vertex].size()));

        // Check if we are already using the requested number of neighbors.
        if (num_neigbors == current_num_neighbors[vertex]) {
            continue;
        }

        vertices_getting_updated.push_back(vertex);

        if (num_neigbors < current_num_neighbors[vertex]) {
            for (int n = num_neigbors; n < current_num_neighbors[vertex]; ++n) {
                const int idx = base_move_indices_involving[vertex][n];
                const MoveGenerator& move = moves[idx];

                assert(is_active_in(move, vertex));
                set_not_active_in(move, vertex);

                vertices_in_updated_moves.insert(move.get_first_vertex());
                vertices_in_updated_moves.insert(move.get_second_vertex());
            }
        } else {
            for (int n = current_num_neighbors[vertex]; n < num_neigbors; ++n) {
                const int idx = base_move_indices_involving[vertex][n];
                const MoveGenerator& move = moves[idx];

                assert(!is_active_in(move, vertex));
                set_active_in(move, vertex);

                vertices_in_updated_moves.insert(move.get_first_vertex());
                vertices_in_updated_moves.insert(move.get_second_vertex());
            }
        }

        current_num_neighbors[vertex] = num_neigbors;
    }

    unique_move_generators.clear();
    unique_endpoints.clear();

    for (const int vertex : vertices_in_updated_moves.get_elements()) {
        unique_move_generators.clear();
        unique_endpoints.clear();

        // We need to scan all base move indices as some movegen may be active due to the other vertex.
        for (int base_idx : base_move_indices_involving[vertex]) {
            assert(base_idx == get_base_move_generator_index(base_idx));
            const auto& move = moves[base_idx];

            if (!is_active_in_any(move)) {
                continue;
            }

            const int idx = vertex == move.get_first_vertex() ? base_idx : get_twin_move_generator_index(base_idx);

            const int j = moves[idx].get_second_vertex();
            if (!unique_endpoints.contains(j)) {
                unique_endpoints.insert_without_checking_existance(j);
                unique_move_generators.push_back(idx);
            }
        }
        active_move_indices_involving_1st[vertex] = unique_move_generators;
    }
}

// ---------------------------------------------------------------------
void MoveGenerators::print_profile() const {
    std::cout << "\n========== MOVEGENERATORS PROFILING (Final Optimised) ==========\n";
    std::cout << "Total constructor time  : " << profile_data.total_constructor_ms << " ms\n";
    std::cout << "Pass 1 (parallel gen)   : " << profile_data.pass1_ms << " ms\n";
    std::cout << "Pass 2 (sort+dedup)     : " << profile_data.pass2_ms << " ms\n";
    std::cout << "Loop 2 (sort per vertex): " << profile_data.loop2_ms << " ms\n";
    std::cout << "----------------------------------------------\n";
    std::cout << "Total moves generated   : " << profile_data.total_moves_generated << "\n";
    std::cout << "Unique base moves       : " << profile_data.unique_base_moves << "\n";
    std::cout << "==============================================\n\n";
}

} // namespace cobra
