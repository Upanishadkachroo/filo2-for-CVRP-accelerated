#ifndef _FILO2_KDTREE_HPP_
#define _FILO2_KDTREE_HPP_

#include <array>
#include <queue>
#include <vector>

namespace cobra {

    class KDTree {
    public:
        KDTree(const std::vector<double>& xcoords, const std::vector<double>& ycoords);
        ~KDTree();

        std::vector<int> GetNearestNeighbors(double x, double y, int k) const;

    private:
        struct Point {
            Point(int index, double x, double y);
            int index;
            std::array<double, 2> coords;
        };

        struct Node {
            Node();
            ~Node();
            int cutdim;
            Node *left, *right;
            std::array<double, 2> lobound, hibound;
            int point_index;
        };
        Node* root = nullptr;

        struct HeapNode {
            HeapNode(int point_index, double distance);
            int point_index;
            double distance;
        };

        struct HeapNodeComparator {
            bool operator()(const HeapNode& a, const HeapNode& b) const;
        };

        using KDTreeHeap = std::priority_queue<HeapNode, std::vector<HeapNode>, HeapNodeComparator>;

        Node* BuildTree(int depth, int begin, int end,
                        const std::array<double, 2>& lobound,
                        const std::array<double, 2>& hibound);

        bool SearchNeighbors(Node* tree, KDTreeHeap& heap,
                             const std::array<double, 2>& point, int k) const;

        bool BoundsOverlapBall(const std::array<double, 2>& point, double dist, Node* node) const;
        bool BallWithinBounds(const std::array<double, 2>& point, double dist, Node* node) const;

        std::vector<Point> nodes;

        static constexpr int TASK_CUTOFF = 500;   // sequential if node span < 500
    };

} // namespace cobra

#endif