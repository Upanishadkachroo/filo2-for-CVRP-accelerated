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


namespace detail {
    inline constexpr double fastround(double value) noexcept {
        return static_cast<int>(value + 0.5);
    }
} // namespace detail

class Instance : private NonCopyable<Instance> {
public:
    static std::optional<Instance> make(const std::string& filepath,
                                        int num_neighbors);



    inline int get_vertices_num() const noexcept {
        return static_cast<int>(demands.size());
    }

    inline int get_depot() const noexcept { return 0; }

    inline int get_vehicle_capacity() const noexcept {
        return vehicle_capacity;
    }

    inline int get_customers_num() const noexcept {
        return get_vertices_num() - 1;
    }

    inline int get_customers_begin() const noexcept { return 1; }

    inline int get_customers_end() const noexcept {
        return get_vertices_num();
    }

    inline int get_vertices_begin() const noexcept { return get_depot(); }

    inline int get_vertices_end() const noexcept {
        return get_customers_end();
    }

    inline double get_cost(int i, int j) const noexcept {
        const double dx = xcoords[i] - xcoords[j];
        const double dy = ycoords[i] - ycoords[j];
        return detail::fastround(std::sqrt(dx * dx + dy * dy));
    }


    inline int get_demand(int i) const noexcept { return demands[i]; }

    inline double get_x_coordinate(int i) const noexcept {
        return xcoords[i];
    }

    inline double get_y_coordinate(int i) const noexcept {
        return ycoords[i];
    }

    inline const std::vector<int>& get_neighbors_of(int i) const noexcept {
        return neighbors[i];
    }

    /// Raw coordinate access — used by instance_gpu.cu for device upload.
    inline const std::vector<double>& get_xcoords() const noexcept {
        return xcoords;
    }

    inline const std::vector<double>& get_ycoords() const noexcept {
        return ycoords;
    }

    void gpu_upload() const;

    /// Release device coordinate arrays.  Call once at program exit.
    void gpu_free() const noexcept;

    /// True if coordinates are currently resident on the device.
    bool gpu_ready() const noexcept;

private:
    // Private constructor — called only from Instance::make()
    Instance(const Parser::Data& data, int neighbors_num);

    int vehicle_capacity;

    std::vector<double> xcoords;
    std::vector<double> ycoords;
    std::vector<int>    demands;
    std::vector<std::vector<int>> neighbors;
};

}

#endif // _FILO2_INSTANCE_HPP_
