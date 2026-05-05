#ifndef _FILO2_INSTANCE_HPP_
#define _FILO2_INSTANCE_HPP_

#include <cassert>
#include <cmath>
#include <optional>
#include <string>
#include <vector>

#include "../base/NonCopyable.hpp"
#include "Parser.hpp"

namespace cobra {

namespace {
    inline double fastround(double value) {
        return static_cast<int>(value + 0.5);
    }
}

class Instance : private NonCopyable<Instance> {
public:
    static std::optional<Instance> make(const std::string& filepath, int num_neighbors);

    inline int get_vertices_num() const {
        return demands.size();
    }

    inline int get_depot() const {
        return 0;
    }

    inline int get_vehicle_capacity() const {
        return vehicle_capacity;
    }

    inline int get_customers_num() const {
        return get_vertices_num() - 1;
    }

    inline int get_customers_begin() const {
        return 1;
    }

    inline int get_customers_end() const {
        return get_vertices_num();
    }

    inline int get_vertices_begin() const {
        return get_depot();
    }

    inline int get_vertices_end() const {
        return get_customers_end();
    }

    inline double get_cost(int i, int j) const {
        const double dx = xcoords[i] - xcoords[j];
        const double dy = ycoords[i] - ycoords[j];
        const double dist = std::sqrt(dx * dx + dy * dy);
        return fastround(dist);
    }

    inline int get_demand(int i) const {
        return demands[i];
    }

    inline double get_x_coordinate(int i) const {
        return xcoords[i];
    }

    inline double get_y_coordinate(int i) const {
        return ycoords[i];
    }

    inline const std::vector<int>& get_neighbors_of(int i) const {
        return neighbors[i];
    }

    // NEW: expose raw coordinate arrays safely
    inline const std::vector<double>& get_xcoords() const {
        return xcoords;
    }

    inline const std::vector<double>& get_ycoords() const {
        return ycoords;
    }

private:
    Instance(const Parser::Data& data, int neighbors_num);

    int vehicle_capacity;

    std::vector<double> xcoords;
    std::vector<double> ycoords;
    std::vector<int> demands;
    std::vector<std::vector<int>> neighbors;
};

}  // namespace cobra

#endif