/**
 * KDTree.hpp  (Phase 1 — batch 2, parallelized)
 *
 * What changed vs the original
 * ----------------------------
 * 1. `GetNearestNeighborsBatch()` — new public method.
 *    Runs all N per-vertex KNN queries in parallel using OpenMP.
 *    Instance.cpp calls this instead of the per-vertex loop.
 *
 * 2. Thread-safety contract documented explicitly (see below).
 *
 * 3. Everything else — Node, Point, HeapNode, BuildTree, SearchNeighbors,
 *    BoundsOverlapBall, BallWithinBounds — is UNCHANGED.
 *
 * Thread-safety contract
 * ----------------------
 * After the constructor returns the KDTree object is IMMUTABLE:
 *   • `nodes`  — flat point array, read-only.
 *   • `root`   — pointer to tree root, read-only.
 *   • Tree node pointers (left/right) — read-only.
 *
 * `GetNearestNeighbors()` is RE-ENTRANT because:
 *   • It only reads the shared tree (immutable after ctor).
 *   • Its `KDTreeHeap heap` is a LOCAL stack variable — one per call.
 *   • It writes only to the returned `std::vector<int>` (caller-owned).
 *
 * Therefore multiple threads can call `GetNearestNeighbors()` simultaneously
 * on the same KDTree object with no locking needed.
 *
 * `BuildTree()` is NOT thread-safe: it mutates `nodes` via std::nth_element.
 * It is only called from the constructor (single-threaded), so this is fine.
 */

#ifndef _FILO2_KDTREE_HPP_
#define _FILO2_KDTREE_HPP_

#include <array>
#include <queue>
#include <vector>

#ifdef _OPENMP
#  include <omp.h>
#endif

namespace cobra {

    // A simple implementation of a kd-tree based on https://github.com/cdalitz/kdtree-cpp.
    class KDTree {
    public:
        KDTree(const std::vector<double>& xcoords, const std::vector<double>& ycoords);
        ~KDTree();

        // Retrieves the k nearest neighbors of point (x, y).
        // RE-ENTRANT: safe to call from multiple threads simultaneously.
        std::vector<int> GetNearestNeighbors(double x, double y, int k) const;

        /**
         * GetNearestNeighborsBatch
         *
         * Fills `out_neighbors[i]` with the k nearest neighbors of vertex i,
         * for all i in [begin, end), in parallel using OpenMP.
         *
         * Parameters
         *   xcoords      — x-coordinates of all vertices (indexed 0..N-1)
         *   ycoords      — y-coordinates of all vertices (indexed 0..N-1)
         *   k            — number of neighbors per vertex (already clamped by caller)
         *   begin        — first vertex index to process (inclusive)
         *   end          — last vertex index to process (exclusive)
         *   out_neighbors — pre-sized output vector[N]; caller must have called
         *                   out_neighbors.resize(N) before this call.
         *
         * Thread safety
         *   Each OpenMP thread queries a disjoint set of vertices.
         *   Each call to GetNearestNeighbors() uses its own stack-local heap.
         *   Writes go to disjoint out_neighbors[i] slots.
         *   No locking required.
         */
        void GetNearestNeighborsBatch(
            const std::vector<double>& xcoords,
            const std::vector<double>& ycoords,
            int k,
            int begin,
            int end,
            std::vector<std::vector<int>>& out_neighbors) const;

    private:
        // A 2D point representation.
        struct Point {
            Point(int index, double x, double y);
            // Point index.
            int index;
            // x and y coordinates.
            std::array<double, 2> coords;
        };

        // A KDTree node representation.
        struct Node {
            Node();
            ~Node();
            // Cut dimension: 0 for the x coordinate, 1 for the y one.
            int cutdim;
            // Child nodes.
            Node *left, *right;
            // Node bounding box.
            std::array<double, 2> lobound, hibound;
            // Index of the point in nodes vector.
            int point_index;
        };
        Node* root = nullptr;

        // Heap node representation.
        struct HeapNode {
            HeapNode(int point_index, double distance);
            // Index of the point in nodes vector.
            int point_index;
            // Distance of this neighbor from the input point.
            double distance;
        };

        // Heap node comparator.
        struct HeapNodeComparator {
            bool operator()(const HeapNode& a, const HeapNode& b) const;
        };

        // Max heap used to retrieve the set of nearest neighbors of a given point.
        using KDTreeHeap = std::priority_queue<HeapNode, std::vector<HeapNode>, HeapNodeComparator>;

        // Builds the tree using nodes indexed from begin to end excluded.
        // NOT thread-safe: mutates `nodes` via std::nth_element.
        // Only called from the constructor (single-threaded).
        Node* BuildTree(int depth, int begin, int end,
                        const std::array<double, 2>& lobound,
                        const std::array<double, 2>& hibound);

        bool SearchNeighbors(Node* tree, KDTreeHeap& heap,
                             const std::array<double, 2>& point, int k) const;

        bool BoundsOverlapBall(const std::array<double, 2>& point, double dist,
                               Node* node) const;
        bool BallWithinBounds(const std::array<double, 2>& point, double dist,
                              KDTree::Node* node) const;

        // Flat point array — READ-ONLY after construction.
        std::vector<Point> nodes;
    };

}  // namespace cobra

#endif  // _FILO2_KDTREE_HPP_