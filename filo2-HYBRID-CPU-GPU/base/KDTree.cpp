/**
 * KDTree.cpp  (Phase 1 — batch 2, parallelized)
 *
 * What changed vs the original
 * ----------------------------
 * ONE new method added at the bottom: GetNearestNeighborsBatch().
 * Every other function body is identical to the original.
 *
 * Why OpenMP and not Thrust/CUDA here
 * ------------------------------------
 * KNN on a pointer-based binary tree is pointer-chasing work — each query
 * follows a different path through heap-allocated Node objects.  GPUs execute
 * efficiently only on regular, coalesced memory access patterns.  Mapping a
 * pointer tree to GPU memory would require:
 *   (a) Linearising all Node* pointers into a flat index array (~300 lines),
 *   (b) Copying the entire tree to device memory for EVERY Instance construction,
 *   (c) Writing a CUDA kernel for tree traversal (complex, cache-unfriendly on GPU).
 *
 * The query loop in Instance.cpp is the bottleneck (N=1M queries, each O(log N)).
 * It is embarrassingly parallel: each query reads the same immutable tree and
 * writes to a private heap.  OpenMP parallel-for gives linear speedup on the
 * number of CPU cores with zero restructuring of the tree itself.
 *
 * Thread safety proof for GetNearestNeighborsBatch
 * -------------------------------------------------
 * Shared read:  `nodes` array — immutable after KDTree ctor returns.
 *               `root` and all Node* pointers — immutable after ctor.
 * Thread-local: `KDTreeHeap heap` — stack-allocated inside GetNearestNeighbors,
 *               one per call, never shared.
 * Disjoint writes: each thread writes only to out_neighbors[i] for its own i.
 * => No mutex, no atomic needed.
 */

#include "KDTree.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <limits>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace cobra {

    // ─────────────────────────────────────────────────────────────────────────
    // Constructor / Destructor — UNCHANGED
    // ─────────────────────────────────────────────────────────────────────────

    KDTree::KDTree(const std::vector<double>& xcoords, const std::vector<double>& ycoords) {

        assert(xcoords.size() == ycoords.size());

        std::array<double, 2> lobound = {std::numeric_limits<double>::max(),
                                          std::numeric_limits<double>::max()};
        std::array<double, 2> hibound = {std::numeric_limits<double>::lowest(),
                                          std::numeric_limits<double>::lowest()};

        for (int i = 0; i < static_cast<int>(xcoords.size()); ++i) {
            lobound[0] = std::min(lobound[0], xcoords[i]);
            lobound[1] = std::min(lobound[1], ycoords[i]);
            hibound[0] = std::max(hibound[0], xcoords[i]);
            hibound[1] = std::max(hibound[1], ycoords[i]);
            nodes.emplace_back(i, xcoords[i], ycoords[i]);
        }

        // BuildTree mutates `nodes` (nth_element) — single-threaded, correct.
        root = BuildTree(0, 0, nodes.size(), lobound, hibound);
    }

    KDTree::~KDTree() {
        delete root;
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Inner type constructors — UNCHANGED
    // ─────────────────────────────────────────────────────────────────────────

    KDTree::Point::Point(int index, double x, double y)
        : index(index), coords({x, y}) { }

    KDTree::Node::Node()  = default;
    KDTree::Node::~Node() {
        delete left;
        delete right;
    }

    KDTree::HeapNode::HeapNode(int point_index, double distance)
        : point_index(point_index), distance(distance) { }

    bool KDTree::HeapNodeComparator::operator()(
            const KDTree::HeapNode& a, const KDTree::HeapNode& b) const {
        return a.distance < b.distance;
    }

    // ─────────────────────────────────────────────────────────────────────────
    // BuildTree — UNCHANGED
    // Mutates `nodes` via std::nth_element. NOT thread-safe.
    // Only called from constructor (single-threaded). ✓
    // ─────────────────────────────────────────────────────────────────────────

    KDTree::Node* KDTree::BuildTree(int depth, int begin, int end,
                                    const std::array<double, 2>& lobound,
                                    const std::array<double, 2>& hibound) {

        const int dimension = depth % 2;

        KDTree::Node* node  = new KDTree::Node();
        node->cutdim        = dimension;
        node->left          = nullptr;
        node->right         = nullptr;
        node->lobound       = lobound;
        node->hibound       = hibound;

        if (end - begin <= 1) {
            node->point_index = begin;
        } else {
            int median = (begin + end) / 2;
            std::nth_element(nodes.begin() + begin,
                             nodes.begin() + median,
                             nodes.begin() + end,
                             [dimension](const Point& a, const Point& b) {
                                 return a.coords[dimension] < b.coords[dimension];
                             });
            node->point_index = median;

            const int cutval = nodes[median].coords[dimension];

            if (median - begin > 0) {
                std::array<double, 2> next_hibound = hibound;
                next_hibound[dimension] = cutval;
                node->left = BuildTree(depth + 1, begin, median,
                                       node->lobound, next_hibound);
            }

            if (end - median > 1) {
                std::array<double, 2> next_lobound = lobound;
                next_lobound[dimension] = cutval;
                node->right = BuildTree(depth + 1, median + 1, end,
                                        next_lobound, hibound);
            }
        }

        return node;
    }

    // ─────────────────────────────────────────────────────────────────────────
    // GetNearestNeighbors — UNCHANGED
    // RE-ENTRANT: heap is local, tree is immutable. ✓
    // ─────────────────────────────────────────────────────────────────────────

    std::vector<int> KDTree::GetNearestNeighbors(double x, double y, int k) const {

        KDTreeHeap heap;                              // private per call
        SearchNeighbors(root, heap, {x, y}, k);

        std::vector<int> neighbors(k);
        while (!heap.empty()) {
            const HeapNode& heap_node = heap.top();
            neighbors[--k] = nodes[heap_node.point_index].index;
            heap.pop();
        }
        return neighbors;
    }

    // ─────────────────────────────────────────────────────────────────────────
    // GetNearestNeighborsBatch  ◀ NEW — OpenMP parallel bulk query
    //
    // Called by Instance.cpp instead of the per-vertex for-loop.
    // Splits the [begin, end) vertex range across all available OpenMP threads.
    // Each thread has its own stack-local `heap` inside GetNearestNeighbors().
    // Writes go to disjoint out_neighbors[i] slots — no synchronization needed.
    // ─────────────────────────────────────────────────────────────────────────

    void KDTree::GetNearestNeighborsBatch(
            const std::vector<double>&    xcoords,
            const std::vector<double>&    ycoords,
            int                           k,
            int                           begin,
            int                           end,
            std::vector<std::vector<int>>& out_neighbors) const
    {
        // schedule(dynamic, 64):
        //   KNN queries have variable cost depending on how many tree nodes
        //   are visited (dense clusters → deeper search).  Dynamic scheduling
        //   with a chunk of 64 vertices balances load across threads without
        //   excessive scheduling overhead.
        #pragma omp parallel for schedule(dynamic, 64) default(none) \
            shared(xcoords, ycoords, k, begin, end, out_neighbors)
        for (int i = begin; i < end; ++i) {

            // GetNearestNeighbors is re-entrant: private heap, reads only
            // the immutable tree. No lock needed.
            out_neighbors[i] = GetNearestNeighbors(xcoords[i], ycoords[i], k);

            // Ensure vertex i is at position 0 in its own neighbor list.
            // (If multiple vertices overlap and k is small, i might not appear.)
            // This swap touches only out_neighbors[i] — disjoint per thread. ✓
            if (out_neighbors[i][0] != i) {
                int n = 1;
                while (n < static_cast<int>(out_neighbors[i].size())) {
                    if (out_neighbors[i][n] == i) { break; }
                    ++n;
                }
                std::swap(out_neighbors[i][0], out_neighbors[i][n]);
            }

            assert(out_neighbors[i][0] == i);
        }
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Distance helpers — UNCHANGED
    // ─────────────────────────────────────────────────────────────────────────

    double ComputeDistance(const std::array<double, 2>& a,
                           const std::array<double, 2>& b) {
        return (a[0] - b[0]) * (a[0] - b[0]) + (a[1] - b[1]) * (a[1] - b[1]);
    }

    double ComputeCoordinateDistance(double a, double b) {
        return (a - b) * (a - b);
    }

    // ─────────────────────────────────────────────────────────────────────────
    // BoundsOverlapBall — UNCHANGED
    // ─────────────────────────────────────────────────────────────────────────

    bool KDTree::BoundsOverlapBall(const std::array<double, 2>& point,
                                   double dist,
                                   KDTree::Node* node) const {
        double distsum = 0;
        for (int i = 0; i < static_cast<int>(point.size()); ++i) {
            if (point[i] < node->lobound[i]) {
                distsum += ComputeCoordinateDistance(point[i], node->lobound[i]);
                if (distsum > dist) return false;
            } else if (point[i] > node->hibound[i]) {
                distsum += ComputeCoordinateDistance(point[i], node->hibound[i]);
                if (distsum > dist) return false;
            }
        }
        return true;
    }

    // ─────────────────────────────────────────────────────────────────────────
    // BallWithinBounds — UNCHANGED
    // ─────────────────────────────────────────────────────────────────────────

    bool KDTree::BallWithinBounds(const std::array<double, 2>& point,
                                  double dist,
                                  KDTree::Node* node) const {
        for (int i = 0; i < static_cast<int>(point.size()); ++i) {
            if (ComputeCoordinateDistance(point[i], node->lobound[i]) <= dist ||
                ComputeCoordinateDistance(point[i], node->hibound[i]) <= dist) {
                return false;
            }
        }
        return true;
    }

    // ─────────────────────────────────────────────────────────────────────────
    // SearchNeighbors — UNCHANGED
    // RE-ENTRANT: `heap` is passed by reference from GetNearestNeighbors,
    // which creates a fresh heap per call. Each thread has its own heap. ✓
    // ─────────────────────────────────────────────────────────────────────────

    bool KDTree::SearchNeighbors(KDTree::Node*   node,
                                 KDTree::KDTreeHeap& heap,
                                 const std::array<double, 2>& point,
                                 int k) const {

        double currdist = ComputeDistance(point, nodes[node->point_index].coords);

        if (static_cast<int>(heap.size()) < k) {
            heap.push(HeapNode(node->point_index, currdist));
        } else if (currdist < heap.top().distance) {
            heap.pop();
            heap.push(HeapNode(node->point_index, currdist));
        }

        if (point[node->cutdim] < nodes[node->point_index].coords[node->cutdim]) {
            if (node->left) {
                if (SearchNeighbors(node->left, heap, point, k)) { return true; }
            }
        } else {
            if (node->right) {
                if (SearchNeighbors(node->right, heap, point, k)) { return true; }
            }
        }

        double dist = static_cast<int>(heap.size()) < k
                    ? std::numeric_limits<double>::max()
                    : heap.top().distance;

        if (point[node->cutdim] < nodes[node->point_index].coords[node->cutdim]) {
            if (node->right && BoundsOverlapBall(point, dist, node->right)) {
                if (SearchNeighbors(node->right, heap, point, k)) { return true; }
            }
        } else {
            if (node->left && BoundsOverlapBall(point, dist, node->left)) {
                if (SearchNeighbors(node->left, heap, point, k)) { return true; }
            }
        }

        if (static_cast<int>(heap.size()) == k) { dist = heap.top().distance; }

        return BallWithinBounds(point, dist, node);
    }

}  // namespace cobra