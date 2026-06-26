#include <chrono>
#include <fstream>
#include "Parameters.hpp"
#include "base/PrettyPrinter.hpp"
#include "base/Timer.hpp"
#include "base/Welford.hpp"
#include "instance/Instance.hpp"
#include "localsearch/LocalSearch.hpp"
#include "movegen/MoveGenerators.hpp"   
#include "opt/RuinAndRecreate.hpp"
#include "opt/SimulatedAnnealing.hpp"
#include "opt/bpp.hpp"
#include "opt/routemin.hpp"
#include "solution/Solution.hpp"
#include "solution/savings.hpp"

#ifdef _OPENMP
#include <omp.h>
#endif

#ifdef GUI
    #include "Renderer.hpp"
#endif

auto get_basename(const std::string& pathname) -> std::string {
    return {std::find_if(pathname.rbegin(), pathname.rend(), [](char c) { return c == '/'; }).base(), pathname.end()};
}

int main(int argc, char* argv[]) {

    cobra::Timer global_timer;
#ifdef VERBOSE
    cobra::Timer timer;
#endif

    const auto params = Parameters(argc, argv);

#ifdef VERBOSE
    std::cout << "Pre-processing the instance.\n";
    timer.reset();
#endif

    auto start = std::chrono::high_resolution_clock::now();
    auto maybe_instance = cobra::Instance::make(params.get_instance_path(), params.get_neighbors_num());
    auto stop = std::chrono::high_resolution_clock::now();
    std::cout << "Loading took " << std::chrono::duration_cast<std::chrono::milliseconds>(stop - start).count() << " ms\n";

#ifdef VERBOSE
    std::cout << "Done in " << timer.elapsed_time<std::chrono::milliseconds>() << " ms.\n\n";
#endif

    if (!maybe_instance.has_value()) {
        return EXIT_FAILURE;
    }

    const cobra::Instance instance = std::move(maybe_instance.value());

    std::cout << "\nINSTANCE LOADED\n";
    std::cout << "\n========== INSTANCE INFO ==========\n";
    std::cout << "Vertices number        : " << instance.get_vertices_num() << std::endl;
    std::cout << "Customers number       : " << instance.get_customers_num() << std::endl;
    std::cout << "Vehicle capacity       : " << instance.get_vehicle_capacity() << std::endl;
    std::cout << "Depot index            : " << instance.get_depot() << std::endl;

    std::cout << "\nCustomers (id, x, y, demand):\n";
    for (int i = instance.get_customers_begin(); i < instance.get_customers_end() && i < 10; ++i) {
        std::cout << "  " << i << " : ("
                  << instance.get_x_coordinate(i) << ", "
                  << instance.get_y_coordinate(i) << ") demand = "
                  << instance.get_demand(i) << std::endl;
    }
    if (instance.get_customers_num() > 10)
        std::cout << "  ... (only first 10 shown)\n";

    std::cout << "\nNumber of stored neighbors for depot: "
              << instance.get_neighbors_of(instance.get_depot()).size() << std::endl;

    if (instance.get_customers_num() > 0) {
        int first_customer = instance.get_customers_begin();
        std::cout << "Cost from depot to customer " << first_customer << " : "
                  << instance.get_cost(instance.get_depot(), first_customer) << std::endl;
    }

    std::cout << "===================================\n\n";

    auto best_solution = cobra::Solution(instance, std::min(instance.get_vertices_num(), params.get_solution_cache_size()));

    auto constr_start = std::chrono::high_resolution_clock::now();
#ifdef VERBOSE
    std::cout << "Running CLARKE&WRIGHT to generate an initial solution.\n";
    timer.reset();
#endif
    cobra::clarke_and_wright(instance, best_solution, params.get_cw_lambda(), params.get_cw_neighbors());
    auto constr_end = std::chrono::high_resolution_clock::now();
    std::cout << "Construction phase (Clarke & Wright) took "
              << std::chrono::duration_cast<std::chrono::milliseconds>(constr_end - constr_start).count()
              << " ms\n";

#ifdef VERBOSE
    std::cout << "Done in " << timer.elapsed_time<std::chrono::seconds>() << " seconds.\n";
    std::cout << "Initial solution: obj = " << best_solution.get_cost() << ", n. of routes = " << best_solution.get_routes_num() << ".\n\n";
#endif

    auto k = params.get_sparsification_rule_neighbors();

#ifdef VERBOSE
    std::cout << "Setting up MOVEGENERATORS data structures.\n";
    timer.reset();
#endif

    auto move_generators = cobra::MoveGenerators(instance, k);

#ifdef VERBOSE
    std::cout << "Done in " << timer.elapsed_time<std::chrono::seconds>() << " seconds.\n";
    const auto tot_arcs = static_cast<unsigned long>(instance.get_vertices_num()) * static_cast<unsigned long>(instance.get_vertices_num());
    const auto move_gen_num = move_generators.size();
    const auto move_gen_perc = 100.0 * static_cast<double>(move_gen_num) / static_cast<double>(tot_arcs);
    std::cout << "Using at most " << move_generators.size() << " move-generators out of " << tot_arcs << " total arcs ";
    std::cout << std::fixed;
    std::cout << std::setprecision(5);
    std::cout << "(approx. " << move_gen_perc << "%)\n\n";
    std::cout << std::defaultfloat;
#endif

#ifdef VERBOSE
    std::cout << "Computing a greedy upper bound on the n. of routes.\n";
    timer.reset();
#endif

    auto kmin = bpp::greedy_first_fit_decreasing(instance);

#ifdef VERBOSE
    std::cout << "Done in " << timer.elapsed_time<std::chrono::milliseconds>() << " milliseconds.\n";
    std::cout << "Around " << kmin << " routes should do the job.\n\n";
#endif

    auto rand_engine = std::mt19937(params.get_seed());
    const auto tolerance = params.get_tolerance();

    // ──────────────────────────────────────────────────────────────────────────────
    //  ROUTEMIN – Parallel restarts (only thread 0 prints)
    // ──────────────────────────────────────────────────────────────────────────────
    if (kmin < best_solution.get_routes_num()) {

        const auto routemin_iterations = params.get_routemin_iterations();

#ifdef VERBOSE
        std::cout << "Running ROUTEMIN heuristic for at most " << routemin_iterations << " iterations.\n";
        std::cout << "Starting solution: obj = " << best_solution.get_cost() << ", n. of routes = " << best_solution.get_routes_num()
                  << ".\n";
        timer.reset();
#endif

#ifdef _OPENMP
        const int num_restarts = std::min(omp_get_max_threads(), 8);
#else
        const int num_restarts = 1;
#endif

        std::vector<uint32_t> thread_seeds(num_restarts);
        for (int t = 0; t < num_restarts; ++t)
            thread_seeds[t] = rand_engine();

        std::vector<cobra::Solution> thread_solutions;
        thread_solutions.reserve(num_restarts);
        for (int t = 0; t < num_restarts; ++t)
            thread_solutions.emplace_back(best_solution);

        std::vector<cobra::MoveGenerators*> thread_mg;
        thread_mg.reserve(num_restarts);
        for (int t = 0; t < num_restarts; ++t)
            thread_mg.push_back(new cobra::MoveGenerators(instance, k, false));

#ifdef _OPENMP
        #pragma omp parallel for num_threads(num_restarts) schedule(static, 1)
#endif
        for (int t = 0; t < num_restarts; ++t) {
            std::mt19937 local_rng(thread_seeds[t]);

            // Suppress stdout for threads other than 0
            if (t == 0) {
                thread_solutions[t] = routemin(
                    instance, best_solution, local_rng,
                    *thread_mg[t], kmin, routemin_iterations, tolerance);
            } else {
                std::streambuf* old_buf = std::cout.rdbuf();
                std::ofstream devnull("/dev/null");
                std::cout.rdbuf(devnull.rdbuf());

                thread_solutions[t] = routemin(
                    instance, best_solution, local_rng,
                    *thread_mg[t], kmin, routemin_iterations, tolerance);

                std::cout.rdbuf(old_buf);
            }
        }

        for (auto* mg : thread_mg) delete mg;
        thread_mg.clear();

        best_solution = thread_solutions[0];
        for (int t = 1; t < num_restarts; ++t) {
            const auto& ts = thread_solutions[t];
            if (ts.get_routes_num() < best_solution.get_routes_num() ||
               (ts.get_routes_num() == best_solution.get_routes_num() &&
                ts.get_cost()       < best_solution.get_cost())) {
                best_solution = ts;
            }
        }

#ifdef VERBOSE
        std::cout << "Final solution: obj = " << best_solution.get_cost() << ", n. routes = " << best_solution.get_routes_num() << "\n";
        std::cout << "Done in " << timer.elapsed_time<std::chrono::seconds>() << " seconds.\n\n";
#endif
    }

    // ──────────────────────────────────────────────────────────────────────────────
    //  COREOPT – Parallel trajectories (only thread 0 prints)
    // ──────────────────────────────────────────────────────────────────────────────

#ifdef TIMELIMIT
    const int optimization_seconds = params.get_optimization_seconds();
    const int optimization_milliseconds = 1000 * optimization_seconds;
#else
    const auto coreopt_iterations = params.get_coreopt_iterations();
#endif

    auto vertices_dist = std::uniform_int_distribution(instance.get_vertices_begin(), instance.get_vertices_end() - 1);
    cobra::Welford welf;
    for (int i = 0; i < instance.get_vertices_num(); ++i) {
        welf.update(instance.get_cost(vertices_dist(rand_engine), vertices_dist(rand_engine)));
    }
    const auto sa_initial_temperature = welf.get_mean() * params.get_sa_initial_factor();
    const auto sa_final_temperature = sa_initial_temperature * params.get_sa_final_factor();

#ifdef VERBOSE
    std::cout << "Simulated annealing temperature goes from " << sa_initial_temperature << " to " << sa_final_temperature << ".\n\n";
#endif

#ifdef _OPENMP
    const int num_threads = std::min(omp_get_max_threads(), 8);
#else
    const int num_threads = 1;
#endif

    std::vector<uint32_t> coreopt_seeds(num_threads);
    for (int t = 0; t < num_threads; ++t)
        coreopt_seeds[t] = rand_engine();

    std::vector<cobra::Solution> thread_solutions(num_threads, best_solution);

    // Build MoveGenerators copies IN PARALLEL (suppress profiling)
    std::vector<cobra::MoveGenerators*> thread_mg(num_threads, nullptr);
#ifdef _OPENMP
    #pragma omp parallel for num_threads(num_threads) schedule(static, 1)
#endif
    for (int t = 0; t < num_threads; ++t) {
        thread_mg[t] = new cobra::MoveGenerators(instance, k, false);
    }

    cobra::Solution global_best = best_solution;

#ifdef _OPENMP
    #pragma omp parallel for num_threads(num_threads) schedule(static, 1)
#endif
    for (int t = 0; t < num_threads; ++t) {
        // Thread-local variables
        std::mt19937 local_rng(coreopt_seeds[t]);
        auto& neighbor = thread_solutions[t];
        cobra::Solution thread_best = neighbor;

        cobra::Timer thread_timer;

#ifdef TIMELIMIT
        int per_thread_time_ms = optimization_milliseconds / num_threads;
        auto sa = cobra::TimeBasedSimulatedAnnealing(sa_initial_temperature, sa_final_temperature, local_rng, per_thread_time_ms);
#else
        int per_thread_iterations = coreopt_iterations / num_threads;
        auto sa = cobra::SimulatedAnnealing(sa_initial_temperature, sa_final_temperature, local_rng, per_thread_iterations);
#endif

        auto rr = RuinAndRecreate(instance, local_rng);

        auto rvnd0 = cobra::RandomizedVariableNeighborhoodDescent(
            instance, *thread_mg[t],
            {cobra::E11, cobra::E10, cobra::TAILS, cobra::SPLIT, cobra::RE22B, cobra::E22, cobra::RE20, cobra::RE21,
             cobra::RE22S, cobra::E21, cobra::E20, cobra::TWOPT, cobra::RE30, cobra::E30, cobra::RE33B, cobra::E33,
             cobra::RE31, cobra::RE32B, cobra::RE33S, cobra::E31, cobra::E32, cobra::RE32S},
            local_rng, tolerance);
        auto rvnd1 = cobra::RandomizedVariableNeighborhoodDescent(instance, *thread_mg[t], {cobra::EJCH}, local_rng, tolerance);
        auto local_search = cobra::VariableNeighborhoodDescentComposer(tolerance);
        local_search.append(&rvnd0);
        local_search.append(&rvnd1);

        const auto gamma_base = params.get_gamma_base();
        auto gamma = std::vector<double>(instance.get_vertices_num(), gamma_base);
        auto gamma_counter = std::vector<int>(instance.get_vertices_num(), 0);
        const auto delta = params.get_delta();
        auto average_number_of_vertices_accessed = cobra::Welford();

        auto gamma_vertices = std::vector<int>();
        for (auto i = instance.get_vertices_begin(); i < instance.get_vertices_end(); i++) {
            gamma_vertices.emplace_back(i);
        }
        thread_mg[t]->set_active_percentage(gamma, gamma_vertices);

        auto ruined_customers = std::vector<int>();

        const auto intensification_lb = params.get_shaking_lb_factor();
        const auto intensification_ub = params.get_shaking_ub_factor();
        auto mean_solution_arc_cost = neighbor.get_cost() / (static_cast<double>(instance.get_customers_num()) +
                                                             2.0 * static_cast<double>(neighbor.get_routes_num()));
        auto shaking_lb_factor = mean_solution_arc_cost * intensification_lb;
        auto shaking_ub_factor = mean_solution_arc_cost * intensification_ub;

        const auto omega_base = std::max(1, static_cast<int>(std::ceil(std::log(instance.get_vertices_num()))));
        auto omega = std::vector<int>(instance.get_vertices_num(), omega_base);
        auto random_choice = std::uniform_int_distribution(0, 1);

        double reference_solution_cost = neighbor.get_cost();

#ifdef VERBOSE
        // Declare all verbose‑related variables at thread scope (only thread 0 will update/print)
        cobra::PrettyPrinter printer({{"%", cobra::PrettyPrinter::Field::Type::REAL, 5, " "},
                                      {"Iterations", cobra::PrettyPrinter::Field::Type::INTEGER, 10, " "},
                                      {"Objective", cobra::PrettyPrinter::Field::Type::INTEGER, 10, " "},
                                      {"Routes", cobra::PrettyPrinter::Field::Type::INTEGER, 6, " "},
                                      {"Iter/s", cobra::PrettyPrinter::Field::Type::REAL, 10, " "},
                                      {"Eta (s)", cobra::PrettyPrinter::Field::Type::REAL, 10, " "},
                                      {"RR (micro)", cobra::PrettyPrinter::Field::Type::REAL, 10, " "},
                                      {"LS (micro)", cobra::PrettyPrinter::Field::Type::REAL, 10, " "},
                                      {"Gamma", cobra::PrettyPrinter::Field::Type::REAL, 5, " "},
                                      {"Omega", cobra::PrettyPrinter::Field::Type::REAL, 6, " "},
                                      {"Temp", cobra::PrettyPrinter::Field::Type::REAL, 6, " "}});
        unsigned long elapsed_minutes = 0;
        cobra::Timer local_timer;
        cobra::Welford welford_rr;
        cobra::Welford welford_ls;
        cobra::Welford welford_rac_before_shaking;
        cobra::Welford welford_rac_after_shaking;
        cobra::Welford welford_local_optima;
        cobra::Welford welford_shaken_solutions;
#endif

#ifdef TIMELIMIT
        int iter = 0;
        while (thread_timer.elapsed_time<std::chrono::milliseconds>() < per_thread_time_ms) {
#else
        for (int iter = 0; iter < per_thread_iterations; ++iter) {
#endif

            neighbor.apply_undo_list1(neighbor);
            neighbor.clear_do_list1();
            neighbor.clear_undo_list1();
            neighbor.clear_svc();

#ifdef VERBOSE
            cobra::Timer rr_timer;
            if (t == 0) {
                if (global_timer.elapsed_time<std::chrono::minutes>() >= elapsed_minutes + 5) {
                    printer.notify("Optimizing for " + std::to_string(global_timer.elapsed_time<std::chrono::minutes>()) + " minutes.");
                    elapsed_minutes += 5;
                }
                rr_timer = cobra::Timer();
            }
#endif

            const auto walk_seed = rr.apply(neighbor, omega);

#ifdef VERBOSE
            if (t == 0) {
                const auto rr_time = rr_timer.elapsed_time<std::chrono::microseconds>();
                welford_rr.update(rr_time);
                welford_rac_after_shaking.update(static_cast<double>(neighbor.get_svc_size()));
                welford_shaken_solutions.update(neighbor.get_cost());
            }
#endif

            ruined_customers.clear();
            for (auto i = neighbor.get_svc_begin(); i != neighbor.get_svc_end(); i = neighbor.get_svc_next(i)) {
                ruined_customers.emplace_back(i);
            }

#ifdef VERBOSE
            cobra::Timer ls_timer;
            if (t == 0) {
                ls_timer = cobra::Timer();
            }
#endif

            local_search.sequential_apply(neighbor);

#ifdef VERBOSE
            if (t == 0) {
                const auto ls_time = ls_timer.elapsed_time<std::chrono::microseconds>();
                welford_ls.update(ls_time);
            }
#endif

            average_number_of_vertices_accessed.update(static_cast<double>(neighbor.get_svc_size()));

#ifdef TIMELIMIT
            double elapsed_sec = thread_timer.elapsed_time<std::chrono::seconds>();
            double avg_iters_sec = (elapsed_sec > 0) ? (iter / elapsed_sec) : 1.0;
            double remaining_sec = (per_thread_time_ms / 1000.0) - elapsed_sec;
            double estimated_total_iters = iter + avg_iters_sec * remaining_sec;
            int max_non_improving_iterations = static_cast<int>(std::ceil(
                delta * estimated_total_iters * average_number_of_vertices_accessed.get_mean() /
                static_cast<double>(instance.get_vertices_num())));
#else
            auto max_non_improving_iterations = static_cast<int>(std::ceil(
                delta * static_cast<double>(per_thread_iterations) *
                average_number_of_vertices_accessed.get_mean() /
                static_cast<double>(instance.get_vertices_num())));
#endif

#ifdef VERBOSE
            if (t == 0) {
                welford_rac_before_shaking.update(static_cast<double>(neighbor.get_svc_size()));
                welford_local_optima.update(neighbor.get_cost());
            }
#endif

            bool improved_thread_best;

            if (neighbor.get_cost() < thread_best.get_cost()) {

                improved_thread_best = true;

                neighbor.apply_do_list2(thread_best);
                neighbor.apply_do_list1(thread_best);
                neighbor.clear_do_list2();

                assert(thread_best == neighbor);

                gamma_vertices.clear();
                for (auto i = neighbor.get_svc_begin(); i != neighbor.get_svc_end(); i = neighbor.get_svc_next(i)) {
                    gamma[i] = gamma_base;
                    gamma_counter[i] = 0;
                    gamma_vertices.emplace_back(i);
                }
                thread_mg[t]->set_active_percentage(gamma, gamma_vertices);

#ifdef VERBOSE
                if (t == 0) {
                    welford_local_optima.reset();
                    welford_local_optima.update(neighbor.get_cost());
                    welford_shaken_solutions.reset();
                    welford_shaken_solutions.update(neighbor.get_cost());
                }
#endif

            } else {

                improved_thread_best = false;

                for (auto i = neighbor.get_svc_begin(); i != neighbor.get_svc_end(); i = neighbor.get_svc_next(i)) {
                    gamma_counter[i]++;
                    if (gamma_counter[i] >= max_non_improving_iterations) {
                        gamma[i] = std::min(gamma[i] * 2.0, 1.0);
                        gamma_counter[i] = 0;
                        gamma_vertices.clear();
                        gamma_vertices.emplace_back(i);
                        thread_mg[t]->set_active_percentage(gamma, gamma_vertices);
                    }
                }
            }

            const auto seed_shake_value = omega[walk_seed];

            if (neighbor.get_cost() > shaking_ub_factor + reference_solution_cost) {
                for (auto i : ruined_customers) {
                    if (omega[i] > seed_shake_value - 1) {
                        omega[i]--;
                    }
                }
            } else if (neighbor.get_cost() >= reference_solution_cost && neighbor.get_cost() < reference_solution_cost + shaking_lb_factor) {
                for (auto i : ruined_customers) {
                    if (omega[i] < seed_shake_value + 1) {
                        omega[i]++;
                    }
                }
            } else {
                for (auto i : ruined_customers) {
                    if (random_choice(local_rng)) {
                        if (omega[i] > seed_shake_value - 1) {
                            omega[i]--;
                        }
                    } else {
                        if (omega[i] < seed_shake_value + 1) {
                            omega[i]++;
                        }
                    }
                }
            }

#ifdef TIMELIMIT
            if (sa.accept(reference_solution_cost, neighbor, thread_timer.elapsed_time<std::chrono::milliseconds>())) {
#else
            if (sa.accept(reference_solution_cost, neighbor)) {
#endif

                if (!improved_thread_best) {
                    neighbor.append_do_list1_to_do_list2();
                }

                neighbor.clear_do_list1();
                neighbor.clear_undo_list1();

                reference_solution_cost = neighbor.get_cost();

                const auto updated_mean_solution_arc_cost = neighbor.get_cost() / (static_cast<double>(instance.get_customers_num()) +
                                                                                   2.0 * static_cast<double>(neighbor.get_routes_num()));
                shaking_lb_factor = updated_mean_solution_arc_cost * intensification_lb;
                shaking_ub_factor = updated_mean_solution_arc_cost * intensification_ub;
            }

            sa.decrease_temperature();

#ifdef GUI
            // GUI disabled in parallel mode
#endif

#ifdef VERBOSE
            if (t == 0 && local_timer.elapsed_time<std::chrono::seconds>() > 1) {
                local_timer.reset();

                auto gamma_mean = 0.0;
                for (auto i = instance.get_vertices_begin(); i < instance.get_vertices_end(); i++) {
                    gamma_mean += gamma[i];
                }
                gamma_mean = (gamma_mean / static_cast<double>(instance.get_vertices_num()));

                auto omega_mean = 0.0;
                for (auto i = instance.get_customers_begin(); i < instance.get_customers_end(); i++) {
                    omega_mean += omega[i];
                }
                omega_mean /= static_cast<double>(instance.get_customers_num());

    #ifdef TIMELIMIT
                const int elapsed_time_ms = thread_timer.elapsed_time<std::chrono::milliseconds>();
                const auto progress = 100.0f * (elapsed_time_ms / 1000.0) / (per_thread_time_ms / 1000.0);
                printer.print(progress, iter + 1, thread_best.get_cost(), thread_best.get_routes_num(),
                              avg_iters_sec, remaining_sec,
                              welford_rr.get_mean(), welford_ls.get_mean(),
                              gamma_mean, omega_mean, sa.get_temperature(elapsed_time_ms));
    #else
                const auto progress = 100.0 * (iter + 1.0) / per_thread_iterations;
                const auto elapsed_seconds = thread_timer.elapsed_time<std::chrono::seconds>();
                const auto iter_per_second = static_cast<double>(iter + 1) / (static_cast<double>(elapsed_seconds) + 0.01);
                const auto remaining_iter = per_thread_iterations - iter;
                const auto estimated_rem_time = static_cast<double>(remaining_iter) / iter_per_second;

                printer.print(progress, iter + 1, thread_best.get_cost(), thread_best.get_routes_num(),
                              iter_per_second, estimated_rem_time,
                              welford_rr.get_mean(), welford_ls.get_mean(),
                              gamma_mean, omega_mean, sa.get_temperature());
    #endif
            }
#endif

#ifdef TIMELIMIT
            ++iter;
        } // while time
#endif
        } // for iterations

        // Update global best
#ifdef _OPENMP
        #pragma omp critical
#endif
        {
            if (thread_best.get_cost() < global_best.get_cost()) {
                global_best = thread_best;
            }
        }
    }

    best_solution = global_best;

    for (auto* mg : thread_mg) delete mg;
    thread_mg.clear();

    // --- Final output (unchanged) ---
    int global_time_elapsed = global_timer.elapsed_time<std::chrono::seconds>();

#ifdef VERBOSE
    std::cout << "\n";
    std::cout << "Best solution found:\n";
    std::cout << "obj = " << best_solution.get_cost() << ", n. routes = " << best_solution.get_routes_num() << "\n";

    std::cout << "\n";
    std::cout << "Run completed in " << global_time_elapsed << " seconds ";
#endif

    const auto outfile = params.get_outpath() + get_basename(params.get_instance_path()) + "_seed-" + std::to_string(params.get_seed()) +
                         ".out";

    std::filesystem::create_directories(params.get_outpath());

    auto out_stream = std::ofstream(outfile);
    out_stream << std::setprecision(10);
    out_stream << best_solution.get_cost() << "\t" << global_time_elapsed << "\n";
    cobra::Solution::store_to_file(
        instance, best_solution,
        params.get_outpath() + get_basename(params.get_instance_path()) + "_seed-" + std::to_string(params.get_seed()) + ".vrp.sol");

#ifdef VERBOSE
    std::cout << "\n";
    std::cout << "Results stored in\n";
    std::cout << " - " << outfile << "\n";
    std::cout << " - "
              << params.get_outpath() + get_basename(params.get_instance_path()) + "_seed-" + std::to_string(params.get_seed()) + ".vrp.sol"
              << "\n";
#endif

    return EXIT_SUCCESS;
}
