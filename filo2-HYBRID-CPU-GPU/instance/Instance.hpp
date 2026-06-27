#ifndef _FILO2_INSTANCE_HPP_
#define _FILO2_INSTANCE_HPP_

#include <cassert>
#include <cmath>
#include <optional>
#include <string>

#include "../base/NonCopyable.hpp"
#include "Parser.hpp"

// Only include DistanceCache if CUDA is enabled
#ifdef USE_CUDA_NEIGHBORS
    #include "../cuda/DistanceCache.hpp"
#else
    // Forward declaration (still needed for the pointer)
    namespace cobra {
        class DistanceCache;
    }
#endif

namespace cobra {
    namespace {
        inline double fastround(double value) {
            return static_cast<int>(value + 0.5);
        }
    }  // namespace

    // Manages a CVRP instance by providing a set of methods to query its properties.
    class Instance : private NonCopyable<Instance> {
    public:
        // Returns an optional containing a properly built instance if the parsing of the input file completes correctly, nullopt otherwise.
        // The parameter `num_neighbors` specifies the number of neighbors that are precomputed for each vertice. These neighbors are then
        // accessibly by using the `get_neighbors_of` method.
        static std::optional<Instance> make(const std::string& filepath, int num_neighbors);

        // Returns the instance size.
        inline int get_vertices_num() const {
            return demands.size();
        }

        // Returns the depot's index.
        inline int get_depot() const {
            return 0;
        }

        // Returns the vehicle capacity.
        inline int get_vehicle_capacity() const {
            return vehicle_capacity;
        };

        // Returns the number of customers.
        inline int get_customers_num() const {
            return get_vertices_num() - 1;
        };

        // Returns the index of the first customer.
        inline int get_customers_begin() const {
            return 1;
        }

        // Returns the index after the last customer.
        inline int get_customers_end() const {
            return get_vertices_num();
        }

        // Returns the index of the first vertex.
        inline int get_vertices_begin() const {
            return get_depot();
        }

        // Returns the index after the last vertex.
        inline int get_vertices_end() const {
            return get_customers_end();
        }

        // Returns the cost of arc (i, j). If a DistanceCache is active for the current thread,
        // it is consulted first; otherwise the distance is computed from coordinates.
        inline double get_cost(int i, int j) const {
            assert(i >= get_vertices_begin() && i < get_vertices_end());
            assert(j >= get_vertices_begin() && j < get_vertices_end());

            // Check if a thread‑local cache is available.
#ifdef USE_CUDA_NEIGHBORS
            if (current_cache) {
                float cached = current_cache->get_distance(i, j);
                if (cached >= 0.0f) return cached;
                // If not found, fall through to the normal computation.
            }
#endif

            const double sqrt = std::sqrt((xcoords[i] - xcoords[j]) * (xcoords[i] - xcoords[j]) +
                                          (ycoords[i] - ycoords[j]) * (ycoords[i] - ycoords[j]));
            assert(static_cast<int>(std::round(sqrt)) == static_cast<int>(fastround(sqrt)));

            return fastround(sqrt);
        }

        // Returns the demand of vertex `i`. The demand is 0 for the depot.
        inline int get_demand(int i) const {
            return demands[i];
        };

        // Returns the x coordinate of vertex `i`.
        inline double get_x_coordinate(int i) const {
            return xcoords[i];
        };

        // Returns the y coordinate of vertex `i`.
        inline double get_y_coordinate(int i) const {
            return ycoords[i];
        };

        // Returns an array of vertices sorted according to increasing cost from `i`. This array always includes `i` in the first
        // position. The total number of elements in the array is defined by the `num_neighbors` parameter in the `Instance` constructor.
        inline const std::vector<int>& get_neighbors_of(int i) const {
            return neighbors[i];
        };

        // Public getters for the coordinate vectors (used by GPU savings kernel and distance cache).
        inline const std::vector<double>& get_xcoords() const { return xcoords; }
        inline const std::vector<double>& get_ycoords() const { return ycoords; }

        // --- Distance cache support ---
        // Sets the thread‑local distance cache. Call this before any local search operation
        // that should use cached distances. Pass nullptr to clear it.
        static void set_distance_cache(DistanceCache* cache);

        // Returns the current thread‑local distance cache (or nullptr if none).
        static DistanceCache* get_distance_cache();

    private:
        Instance(const Parser::Data& data, int neighbors_num);

        // Maximum vehicle capacity.
        int vehicle_capacity;

        // Vertices x coordinates.
        std::vector<double> xcoords;

        // Vertices y coordinates.
        std::vector<double> ycoords;

        // Vertices demands.
        std::vector<int> demands;

        // Neighbors for each vertex.
        std::vector<std::vector<int>> neighbors;

        // Thread‑local pointer to the distance cache for the current thread.
        // Defined in Instance.cpp.
        static thread_local DistanceCache* current_cache;

        // --- GPU data (to be added for distance cache) ---
#ifdef USE_CUDA_NEIGHBORS
        // Device pointers for coordinates and flattened neighbor lists.
        float* d_xcoords;
        float* d_ycoords;
        int*   d_neighbors_flat;

    public:
        // Getters for GPU data (used by DistanceCache)
        inline const float* get_xcoords_gpu() const { return d_xcoords; }
        inline const float* get_ycoords_gpu() const { return d_ycoords; }
        inline const int*   get_neighbors_flat_gpu() const { return d_neighbors_flat; }
#endif
    };

}  // namespace cobra

#endif
