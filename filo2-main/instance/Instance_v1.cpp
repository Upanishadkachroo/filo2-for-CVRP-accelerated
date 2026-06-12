#include "Instance.hpp"
#include <algorithm>
#include "../base/KDTree.hpp"
#include "../base/Timer.hpp"

#ifdef VERBOSE
#include <iostream>
#endif

#ifdef _OPENMP
#include <omp.h>
#endif

namespace cobra {

    std::optional<Instance> Instance::make(const std::string& filepath, int neighbors_num) {
        Parser parser(filepath);
        std::optional<Parser::Data> maybe_data = parser.Parse();
        if (!maybe_data.has_value()) {
            return std::nullopt;
        }
        return Instance(maybe_data.value(), neighbors_num);
    }

    Instance::Instance(const Parser::Data& data, int neighbors_num) {
        neighbors_num = std::min(neighbors_num, static_cast<int>(data.demands.size()));

        // Move data from parser (O(1), not parallelized)
        vehicle_capacity = data.vehicle_capacity;
        xcoords = std::move(data.xcoords);
        ycoords = std::move(data.ycoords);
        demands = std::move(data.demands);

        neighbors.resize(get_vertices_num());

        // Build KD‑tree (parallelized inside KDTree constructor)
        KDTree kd_tree(xcoords, ycoords);

#ifdef VERBOSE
        Timer timer;
        int last_progress = -1;
#endif

        // Parallelize the neighbor generation loop
        #pragma omp parallel for schedule(dynamic)
        for (int i = get_vertices_begin(); i < get_vertices_end(); ++i) {
            neighbors[i] = kd_tree.GetNearestNeighbors(xcoords[i], ycoords[i], neighbors_num);

            // Ensure the vertex itself is the first entry in its neighbor list
            if (neighbors[i][0] != i) {
                int n = 1;
                while (n < static_cast<int>(neighbors[i].size())) {
                    if (neighbors[i][n] == i) break;
                    ++n;
                }
                std::swap(neighbors[i][0], neighbors[i][n]);
            }

#ifdef VERBOSE
            // Thread‑safe progress reporting (every 10 seconds in milliseconds)
            #pragma omp critical
            {
                int progress = 100 * (i + 1) / get_vertices_num();
                if (timer.elapsed_time<std::chrono::milliseconds>() > 10000 && progress != last_progress) {
                    std::cout << "Progress: " << progress << "%\n";
                    timer.reset();
                    last_progress = progress;
                }
            }
#endif
        }
    }

} // namespace cobra