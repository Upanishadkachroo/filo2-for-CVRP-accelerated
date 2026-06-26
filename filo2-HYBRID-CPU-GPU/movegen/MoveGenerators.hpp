#ifndef _FILO2_MOVEGENERATORS_HPP_
#define _FILO2_MOVEGENERATORS_HPP_

#ifdef _OPENMP
#include <omp.h>
#endif

#if defined(__GNUC__) && defined(_OPENMP)
#include <parallel/algorithm>
#endif

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <vector>
#include <chrono>
#include <iostream>

#include "../base/BinaryHeap.hpp"
#include "../base/Flat2DVector.hpp"
#include "../base/SparseIntSet.hpp"
#include "../base/VectorView.hpp"
#include "../instance/Instance.hpp"

namespace cobra {

    // ---------- TimestampGenerator ----------
    class TimestampGenerator : private NonCopyable<TimestampGenerator> {
    public:
        TimestampGenerator() = default;
        inline unsigned long get() const { return value; }
        inline void increment() { ++value; }
    private:
        unsigned long value = 0;
    };

    // ---------- MoveGenerator ----------
    class MoveGenerator : private NonCopyable<MoveGenerator> {
    public:
        MoveGenerator(int i, int j) : i(i), j(j) { }
        inline int get_first_vertex() const { return i; }
        inline int get_second_vertex() const { return j; }
        inline double get_delta() const { return delta; }
        inline void set_delta(double value) { delta = value; }
        inline int get_heap_index() const { return heap_index; }
        inline void set_heap_index(int index) { heap_index = index; }
        inline bool is_computed_for_ejch() const { return computed_for_ejch; }
        inline void set_computed_for_ejch(bool value) { computed_for_ejch = value; }
    private:
        int i, j;
        double delta = 0.0;
        int heap_index = -1;
        bool computed_for_ejch = false;
    };

    struct MGCompare {
        auto operator()(MoveGenerator* mg1, MoveGenerator* mg2) {
            assert(mg1 && mg2);
            return mg1->get_delta() - mg2->get_delta();
        }
    };

    struct MGGetIdx {
        auto operator()(MoveGenerator* mg1) {
            assert(mg1);
            return mg1->get_heap_index();
        }
    };

    struct MGSetIdx {
        void operator()(MoveGenerator* mg1, int idx) {
            assert(mg1);
            mg1->set_heap_index(idx);
        }
    };

    struct MGUpdate {
        auto operator()(MoveGenerator* mg1, double delta) {
            assert(mg1);
            const auto res = mg1->get_delta() - delta;
            mg1->set_delta(delta);
            return res;
        }
    };

    // ---------- MoveGeneratorsHeap ----------
    class MoveGeneratorsHeap
        : private NonCopyable<MoveGeneratorsHeap>
        , private BinaryHeap<MoveGenerator*, MGCompare, MGGetIdx, MGSetIdx, MGUpdate, -1> {
        using BHeap = BinaryHeap<MoveGenerator*, MGCompare, MGGetIdx, MGSetIdx, MGUpdate>;
    public:
        MoveGeneratorsHeap() = default;
        void reset() { BHeap::reset(); }
        bool is_empty() const { return BHeap::empty(); }
        void insert(MoveGenerator* mg) { BHeap::insert(mg); }
        MoveGenerator* get() { return BHeap::get(); }
        void remove(int heap_index) { BHeap::remove(heap_index); }
        void change_value(int heap_index, double value) { BHeap::update(heap_index, value); }
        int size() const { return BHeap::size(); }
        MoveGenerator* spy(int heap_index) { return BHeap::spy(heap_index); }
        static const int unheaped = -1;
    private:
        void dump() override {
            for (int n = 0; n < size(); ++n) {
                const auto& move = spy(n);
                std::cout << "[" << n << "] (" << move->get_first_vertex() << ", "
                          << move->get_second_vertex() << ") delta = " << move->get_delta()
                          << " heap index = " << move->get_heap_index() << "\n";
            }
        }
    };

    // ---------- MoveGenerators ----------
    class MoveGenerators : private NonCopyable<MoveGenerators> {
    public:
        // ★★★ ADDED: print_profiling parameter (default true) ★★★
        MoveGenerators(const Instance& instance, int k, bool print_profiling = true)
            : max_num_neighbors(std::min(k, instance.get_vertices_num() - 1))
            , heap(MoveGeneratorsHeap())
            , vertex_timestamp(instance.get_vertices_num(), 0)
            , vertices_in_updated_moves(instance.get_vertices_num())
            , unique_endpoints(instance.get_vertices_num()) {

            auto t_construct_start = std::chrono::high_resolution_clock::now();

            const int N = instance.get_vertices_num();
            const int neighbors_begin = 1;
            const int neighbors_end   = neighbors_begin + max_num_neighbors;

#ifdef _OPENMP
            const int num_threads = omp_get_max_threads();
            if (print_profiling) {
                std::cout << "[MoveGenerators] OpenMP enabled, max threads = " << num_threads << "\n";
            }
#else
            const int num_threads = 1;
            if (print_profiling) {
                std::cout << "[MoveGenerators] OpenMP disabled, using sequential execution.\n";
            }
#endif

            struct ThreadLocalData {
                std::vector<int> first;
                std::vector<int> second;
                std::vector<int> cost;
            };

            std::vector<ThreadLocalData> thread_data(num_threads);
            const int chunk_size = (N + num_threads - 1) / num_threads;

            auto t_pass1_start = std::chrono::high_resolution_clock::now();

#ifdef _OPENMP
            #pragma omp parallel for schedule(static)
#endif
            for (int t = 0; t < num_threads; ++t) {
                int start = t * chunk_size;
                int end   = std::min(start + chunk_size, N);
                if (start >= end) continue;

                auto& local = thread_data[t];
                size_t est_moves = (end - start) * max_num_neighbors / 2;
                local.first.reserve(est_moves);
                local.second.reserve(est_moves);
                local.cost.reserve(est_moves);

                auto add_move = [&](int a, int b, int c) {
                    local.first.push_back(a);
                    local.second.push_back(b);
                    local.cost.push_back(c);
                };

                for (int i = start; i < end; ++i) {
                    const auto& ineighbors = instance.get_neighbors_of(i);
                    for (int p = neighbors_begin; p < neighbors_end; ++p) {
                        assert(p < static_cast<int>(ineighbors.size()));
                        const int j = ineighbors[p];
                        if (j < 0) continue;

                        const int cost = static_cast<int>(instance.get_cost(i, j));
                        assert(i != j);

                        if (i < j) {
                            add_move(i, j, cost);
                            continue;
                        }

                        const auto& jneighbors = instance.get_neighbors_of(j);
                        const int cij = cost;
                        const int cjn = static_cast<int>(instance.get_cost(j, jneighbors[neighbors_end - 1]));

                        if (cij > cjn) {
                            add_move(j, i, cost);
                        } else if (cij == cjn) {
                            add_move(j, i, cost);
                        }
                    }
                }
            }

            auto t_pass1_end = std::chrono::high_resolution_clock::now();

            auto t_pass2_start = std::chrono::high_resolution_clock::now();

            size_t total_moves = 0;
            for (int t = 0; t < num_threads; ++t) total_moves += thread_data[t].first.size();

            if (print_profiling) {
                std::cout << "[MoveGenerators] Total candidate moves before sorting: " << total_moves << "\n";
            }

            struct Pair {
                int a, b, cost;
                bool operator<(const Pair& other) const {
                    if (a != other.a) return a < other.a;
                    return b < other.b;
                }
                bool operator==(const Pair& other) const {
                    return a == other.a && b == other.b;
                }
            };

            std::vector<Pair> all_pairs;
            all_pairs.reserve(total_moves);

            for (int t = 0; t < num_threads; ++t) {
                auto& local = thread_data[t];
                for (size_t idx = 0; idx < local.first.size(); ++idx) {
                    int a = local.first[idx];
                    int b = local.second[idx];
                    if (a > b) std::swap(a, b);
                    all_pairs.push_back({a, b, local.cost[idx]});
                }
            }

#if defined(__GNUC__) && defined(_OPENMP)
            if (print_profiling) {
                std::cout << "[MoveGenerators] Using __gnu_parallel::sort (parallel)\n";
            }
            __gnu_parallel::sort(all_pairs.begin(), all_pairs.end(),
                                 [](const Pair& p1, const Pair& p2) { return p1 < p2; });
#else
            if (print_profiling) {
                std::cout << "[MoveGenerators] Using std::sort (sequential)\n";
            }
            std::sort(all_pairs.begin(), all_pairs.end());
#endif

            std::vector<int> vertex_count(N, 0);
            size_t unique_count = 0;
            for (size_t idx = 0; idx < all_pairs.size(); ++idx) {
                if (idx > 0 && all_pairs[idx] == all_pairs[idx-1]) continue;
                ++unique_count;
                const auto& p = all_pairs[idx];
                vertex_count[p.a]++;
                vertex_count[p.b]++;
            }

            moves.reserve(unique_count * 2);
            edge_costs.reserve(unique_count);
            base_move_indices_involving.resize(N);
            for (int i = 0; i < N; ++i) {
                base_move_indices_involving[i].reserve(vertex_count[i]);
            }

            for (size_t idx = 0; idx < all_pairs.size(); ++idx) {
                if (idx > 0 && all_pairs[idx] == all_pairs[idx-1]) continue;
                const auto& p = all_pairs[idx];
                const int base_idx = static_cast<int>(moves.size());
                assert(base_idx == get_base_move_generator_index(base_idx));
                moves.emplace_back(p.a, p.b);
                moves.emplace_back(p.b, p.a);
                edge_costs.emplace_back(static_cast<double>(p.cost));
                base_move_indices_involving[p.a].push_back(base_idx);
                base_move_indices_involving[p.b].push_back(base_idx);
            }

            all_pairs.clear();
            all_pairs.shrink_to_fit();
            thread_data.clear();
            thread_data.shrink_to_fit();

            auto t_pass2_end = std::chrono::high_resolution_clock::now();

            auto t_loop2_start = std::chrono::high_resolution_clock::now();

#ifdef _OPENMP
            #pragma omp parallel for schedule(dynamic, 64)
#endif
            for (int i = 0; i < N; ++i) {
                std::sort(base_move_indices_involving[i].begin(),
                          base_move_indices_involving[i].end(),
                          [this](int a, int b) {
                              return get_edge_cost(get(a)) < get_edge_cost(get(b));
                          });
            }

            auto t_loop2_end = std::chrono::high_resolution_clock::now();

            move_active_in_1st.resize(moves.size() / 2, false);
            move_active_in_2nd.resize(moves.size() / 2, false);

            update_bits.resize(N, 2);
            current_num_neighbors.resize(N, 0);
            active_move_indices_involving_1st.resize(N);

            auto t_construct_end = std::chrono::high_resolution_clock::now();

            // ★★★ Only print if requested ★★★
            if (print_profiling) {
                auto ms = [](auto a, auto b) {
                    return std::chrono::duration_cast<std::chrono::milliseconds>(b - a).count();
                };

                std::cout << "\n========== MOVEGENERATORS PROFILING (Final Optimised) ==========\n";
                std::cout << "Total constructor time  : " << ms(t_construct_start, t_construct_end) << " ms\n";
                std::cout << "Pass 1 (parallel gen)   : " << ms(t_pass1_start, t_pass1_end)         << " ms\n";
                std::cout << "Pass 2 (sort+dedup)     : " << ms(t_pass2_start, t_pass2_end)         << " ms\n";
                std::cout << "Loop 2 (sort per vertex): " << ms(t_loop2_start, t_loop2_end)          << " ms\n";
                std::cout << "----------------------------------------------\n";
                std::cout << "Total moves generated   : " << total_moves                           << "\n";
                std::cout << "Unique base moves       : " << moves.size() / 2                      << "\n";
                std::cout << "==============================================\n\n";
            }
        }

        // ---------- Public methods (unchanged) ----------
        inline MoveGenerator& get(int idx) {
            assert(idx >= 0 && idx < static_cast<int>(moves.size()));
            return moves[idx];
        }
        inline const MoveGenerator& get(int idx) const {
            assert(idx >= 0 && idx < static_cast<int>(moves.size()));
            return moves[idx];
        }

        inline const auto& get_move_generator_indices_involving_1st(int vertex) const {
            return active_move_indices_involving_1st[vertex];
        }

        inline auto get_move_generator_indices_involving_2nd(int vertex) const {
            const auto& v = active_move_indices_involving_1st[vertex];
            return VectorView<decltype(v.begin()), twin_functor>(v.begin(), v.end());
        }

        inline auto get_move_generator_indices_involving(int vertex) const {
            const auto& v = active_move_indices_involving_1st[vertex];
            return VectorView<decltype(v.begin()), base_functor>(v.begin(), v.end());
        }

        void set_active_percentage(std::vector<double>& percentage, std::vector<int>& vertices) {
            vertices_getting_updated.clear();
            vertices_in_updated_moves.clear();

            for (int vertex : vertices) {
                const int num_neigbors = std::round(percentage[vertex] * max_num_neighbors);
                assert(num_neigbors <= static_cast<int>(base_move_indices_involving[vertex].size()));

                if (num_neigbors == current_num_neighbors[vertex]) continue;

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

            for (int vertex : vertices_in_updated_moves.get_elements()) {
                unique_move_generators.clear();
                unique_endpoints.clear();

                for (int base_idx : base_move_indices_involving[vertex]) {
                    assert(base_idx == get_base_move_generator_index(base_idx));
                    const auto& move = moves[base_idx];
                    if (!is_active_in_any(move)) continue;

                    const int idx = (vertex == move.get_first_vertex()) ? base_idx : get_twin_move_generator_index(base_idx);
                    const int j = moves[idx].get_second_vertex();
                    if (!unique_endpoints.contains(j)) {
                        unique_endpoints.insert_without_checking_existance(j);
                        unique_move_generators.push_back(idx);
                    }
                }
                active_move_indices_involving_1st[vertex] = unique_move_generators;
            }
        }

        inline MoveGeneratorsHeap& get_heap() { return heap; }
        inline TimestampGenerator& get_timestamp_generator() { return timegen; }
        inline std::vector<unsigned long>& get_vertex_timestamp() { return vertex_timestamp; }
        inline Flat2DVector<bool>& get_update_bits() { return update_bits; }

        inline double get_edge_cost(const MoveGenerator& move) const {
            return edge_costs[(&move - moves.data()) / 2];
        }

        inline size_t size() const { return moves.size(); }

        static inline int get_twin_move_generator_index(int index) { return index ^ 1; }
        static inline int get_base_move_generator_index(int index) { return index & ~1; }

    private:
        inline void set_active_in(const MoveGenerator& move, int vertex) {
            const int idx = (&move - moves.data()) / 2;
            assert(vertex == move.get_first_vertex() || vertex == move.get_second_vertex());
            if (vertex == move.get_first_vertex()) move_active_in_1st[idx] = true;
            else move_active_in_2nd[idx] = true;
        }

        inline void set_not_active_in(const MoveGenerator& move, int vertex) {
            const int idx = (&move - moves.data()) / 2;
            assert(vertex == move.get_first_vertex() || vertex == move.get_second_vertex());
            if (vertex == move.get_first_vertex()) move_active_in_1st[idx] = false;
            else move_active_in_2nd[idx] = false;
        }

        inline bool is_active_in(const MoveGenerator& move, int vertex) const {
            const int idx = (&move - moves.data()) / 2;
            assert(vertex == move.get_first_vertex() || vertex == move.get_second_vertex());
            return vertex == move.get_first_vertex() ? move_active_in_1st[idx] : move_active_in_2nd[idx];
        }

        inline bool is_active_in_any(const MoveGenerator& move) const {
            const int idx = (&move - moves.data()) / 2;
            return move_active_in_1st[idx] || move_active_in_2nd[idx];
        }

        // ---- Members ----
        const int max_num_neighbors;
        std::vector<MoveGenerator> moves;
        std::vector<std::vector<int>> base_move_indices_involving;
        std::vector<std::vector<int>> active_move_indices_involving_1st;
        std::vector<double> edge_costs;
        std::vector<int> current_num_neighbors;
        std::vector<bool> move_active_in_1st;
        std::vector<bool> move_active_in_2nd;
        MoveGeneratorsHeap heap;
        Flat2DVector<bool> update_bits;
        std::vector<unsigned long> vertex_timestamp;
        TimestampGenerator timegen;

        std::vector<int> vertices_getting_updated;
        SparseIntSet vertices_in_updated_moves;
        std::vector<int> unique_move_generators;
        SparseIntSet unique_endpoints;
    };

} // namespace cobra

#endif
