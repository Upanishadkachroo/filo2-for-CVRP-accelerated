#ifndef _FILO2_MOVEGENERATORS_HPP_
#define _FILO2_MOVEGENERATORS_HPP_

#include <algorithm>
#include <cmath>
#include <vector>
#include <cassert>
#include <iostream>

#include "../base/BinaryHeap.hpp"
#include "../base/Flat2DVector.hpp"
#include "../base/SparseIntSet.hpp"
#include "../base/VectorView.hpp"
#include "../instance/Instance.hpp"

namespace cobra {

// ------------------------------------------------------------
// Timestamp generator
// ------------------------------------------------------------
class TimestampGenerator : private NonCopyable<TimestampGenerator> {
public:
    TimestampGenerator() = default;

    inline unsigned long get() const noexcept {
        return value;
    }

    inline void increment() noexcept {
        ++value;
    }

private:
    unsigned long value = 0;
};

// ------------------------------------------------------------
// MoveGenerator
// ------------------------------------------------------------
class MoveGenerator : private NonCopyable<MoveGenerator> {
public:
    MoveGenerator(int i, int j) noexcept : i(i), j(j) {}

    inline int get_first_vertex() const noexcept { return i; }
    inline int get_second_vertex() const noexcept { return j; }

    inline double get_delta() const noexcept { return delta; }
    inline void set_delta(double value) noexcept { delta = value; }

    inline int get_heap_index() const noexcept { return heap_index; }
    inline void set_heap_index(int index) noexcept { heap_index = index; }

    inline bool is_computed_for_ejch() const noexcept { return computed_for_ejch; }
    inline void set_computed_for_ejch(bool value) noexcept { computed_for_ejch = value; }

private:
    int i;
    int j;
    double delta = 0.0;
    int heap_index = -1;
    bool computed_for_ejch = false;
};

// ------------------------------------------------------------
// Heap helpers
// ------------------------------------------------------------
struct MGCompare {
    bool operator()(MoveGenerator* a, MoveGenerator* b) const noexcept {
        assert(a && b);
        return a->get_delta() < b->get_delta(); // MIN heap
    }
};

struct MGGetIdx {
    int operator()(MoveGenerator* mg) const noexcept {
        assert(mg);
        return mg->get_heap_index();
    }
};

struct MGSetIdx {
    void operator()(MoveGenerator* mg, int idx) const noexcept {
        assert(mg);
        mg->set_heap_index(idx);
    }
};

struct MGUpdate {
    double operator()(MoveGenerator* mg, double delta) const noexcept {
        assert(mg);
        double old = mg->get_delta();
        mg->set_delta(delta);
        return old - delta;
    }
};

// ------------------------------------------------------------
// Heap wrapper
// ------------------------------------------------------------
class MoveGeneratorsHeap
    : private NonCopyable<MoveGeneratorsHeap>
    , private BinaryHeap<MoveGenerator*, MGCompare, MGGetIdx, MGSetIdx, MGUpdate, -1>
{
    using BHeap = BinaryHeap<MoveGenerator*, MGCompare, MGGetIdx, MGSetIdx, MGUpdate>;

public:
    MoveGeneratorsHeap() = default;

    void reset() { BHeap::reset(); }
    bool is_empty() const { return BHeap::empty(); }

    void insert(MoveGenerator* mg) { BHeap::insert(mg); }
    MoveGenerator* get() { return BHeap::get(); }

    void remove(int idx) { BHeap::remove(idx); }
    void change_value(int idx, double val) { BHeap::update(idx, val); }

    int size() const { return BHeap::size(); }

    MoveGenerator* spy(int idx) { return BHeap::spy(idx); }

    static constexpr int unheaped = -1;

private:
    void dump() override {
        for (int i = 0; i < size(); i++) {
            auto* m = spy(i);
            std::cout << "[" << i << "] ("
                      << m->get_first_vertex() << ", "
                      << m->get_second_vertex()
                      << ") delta=" << m->get_delta()
                      << " idx=" << m->get_heap_index()
                      << "\n";
        }
    }
};

// ------------------------------------------------------------
// MoveGenerators (k-NN based)
// ------------------------------------------------------------
class MoveGenerators : private NonCopyable<MoveGenerators> {
public:
    MoveGenerators(const Instance& instance, int k)
        : max_num_neighbors(std::min(k, instance.get_vertices_num() - 1)),
          vertex_timestamp(instance.get_vertices_num(), 0),
          vertices_in_updated_moves(instance.get_vertices_num()),
          unique_endpoints(instance.get_vertices_num())
    {
        update_bits.resize(instance.get_vertices_num(), 2);

        base_move_indices_involving.resize(instance.get_vertices_num());
        active_move_indices_involving_1st.resize(instance.get_vertices_num());
        current_num_neighbors.resize(instance.get_vertices_num(), 0);

        build_base_moves(instance);
    }

    // --------------------------------------------------------
    // Access
    // --------------------------------------------------------
    inline MoveGenerator& get(int idx) noexcept {
        return moves[idx];
    }

    inline const MoveGenerator& get(int idx) const noexcept {
        return moves[idx];
    }

    inline size_t size() const noexcept {
        return moves.size();
    }

    // --------------------------------------------------------
    // Index helpers
    // --------------------------------------------------------
    static inline int get_twin_move_generator_index(int idx) noexcept {
        return idx ^ 1;
    }

    static inline int get_base_move_generator_index(int idx) noexcept {
        return idx & ~1;
    }

    // --------------------------------------------------------
    // Heap access
    // --------------------------------------------------------
    inline MoveGeneratorsHeap& get_heap() noexcept {
        return heap;
    }

    // --------------------------------------------------------
    // Edge cost
    // --------------------------------------------------------
    inline double get_edge_cost(int idx) const noexcept {
        return edge_costs[idx / 2];
    }

    // --------------------------------------------------------
    // Timestamp
    // --------------------------------------------------------
    inline TimestampGenerator& get_timestamp_generator() noexcept {
        return timegen;
    }

    inline std::vector<unsigned long>& get_vertex_timestamp() noexcept {
        return vertex_timestamp;
    }

    inline Flat2DVector<bool>& get_update_bits() noexcept {
        return update_bits;
    }

private:
    // --------------------------------------------------------
    // Build k-NN base graph
    // --------------------------------------------------------
    void build_base_moves(const Instance& instance)
    {
        int n = instance.get_vertices_num();
        int begin = instance.get_vertices_begin();
        int end   = instance.get_vertices_end();

        for (int i = begin; i < end; ++i) {
            const auto& neigh = instance.get_neighbors_of(i);

            for (int p = 1; p <= max_num_neighbors; ++p) {
                int j = neigh[p];
                double cost = instance.get_cost(i, j);

                if (i < j)
                    insert_move(i, j, cost);
                else
                    insert_move(j, i, cost);
            }
        }

        // sort by cost
        for (int i = 0; i < n; ++i) {
            auto& vec = base_move_indices_involving[i];
            std::sort(vec.begin(), vec.end(),
                [this](int a, int b) {
                    return get_edge_cost(a) < get_edge_cost(b);
                });
        }

        move_active_in_1st.resize(moves.size() / 2, false);
        move_active_in_2nd.resize(moves.size() / 2, false);
    }

    // --------------------------------------------------------
    // Insert move pair (i,j) & (j,i)
    // --------------------------------------------------------
    void insert_move(int a, int b, double cost)
    {
        int base_idx = moves.size();

        moves.emplace_back(a, b);
        moves.emplace_back(b, a);

        edge_costs.push_back(cost);

        base_move_indices_involving[a].push_back(base_idx);
        base_move_indices_involving[b].push_back(base_idx);
    }

private:
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