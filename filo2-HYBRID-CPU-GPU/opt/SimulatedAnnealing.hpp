#ifndef _FILO2_SIMULATEDANNEALING_HPP_
#define _FILO2_SIMULATEDANNEALING_HPP_

#include <cmath>
#include <random>
#include <algorithm>
#include <limits>

#include "../solution/Solution.hpp"

namespace cobra {

#ifdef TIMELIMIT


class TimeBasedSimulatedAnnealing {
public:
    TimeBasedSimulatedAnnealing(
        double initial_temperature_,
        double final_temperature_,
        std::mt19937 &rand_engine_,
        int time_ms)
        : rand_engine(rand_engine_),
          uniform_dist(1e-12, 1.0)  // avoid log(0)
    {
        initial_temperature = initial_temperature_;
        final_temperature = final_temperature_;
        period = time_ms;

        temp_ratio = final_temperature / initial_temperature;
    }

    inline double get_temperature(int elapsed_time_ms) const noexcept {
        return initial_temperature *
               std::pow(temp_ratio,
                        static_cast<double>(elapsed_time_ms) / period);
    }

    bool accept(double current_cost,
                const Solution &neighbor,
                int elapsed_time_ms)
    {
        double T = get_temperature(elapsed_time_ms);

        double delta = neighbor.get_cost() - current_cost;

        // Always accept improvement
        if (delta < 0) return true;

        // Avoid zero temperature
        if (T <= 1e-12) return false;

        double prob = std::exp(-delta / T);
        return uniform_dist(rand_engine) < prob;
    }

    void decrease_temperature() {}

private:
    double initial_temperature;
    double final_temperature;
    double temp_ratio;
    int period;

    std::mt19937 &rand_engine;
    std::uniform_real_distribution<double> uniform_dist;
};

#else

// Iteration-based Simulated Annealing
class SimulatedAnnealing {
public:
    SimulatedAnnealing(
        double initial_temperature_,
        double final_temperature_,
        std::mt19937 &rand_engine_,
        int max_iter)
        : rand_engine(rand_engine_),
          uniform_dist(1e-12, 1.0)  // avoid log(0)
    {
        initial_temperature = initial_temperature_;
        final_temperature = final_temperature_;
        period = max_iter;

        temperature = initial_temperature;

        factor = std::pow(final_temperature / initial_temperature,
                          1.0 / static_cast<double>(period));
    }

    inline double get_temperature() const noexcept {
        return temperature;
    }

    inline void decrease_temperature() noexcept {
        temperature = std::max(temperature * factor, 1e-12);
    }

    bool accept(double current_cost, const Solution &neighbor)
    {
        double delta = neighbor.get_cost() - current_cost;

        // Always accept improvement
        if (delta < 0) return true;

        if (temperature <= 1e-12) return false;

        double prob = std::exp(-delta / temperature);

        return uniform_dist(rand_engine) < prob;
    }

private:
    double initial_temperature;
    double final_temperature;
    double temperature;
    int period;

    std::mt19937 &rand_engine;
    std::uniform_real_distribution<double> uniform_dist;

    double factor;
};

#endif

} // namespace cobra

#endif