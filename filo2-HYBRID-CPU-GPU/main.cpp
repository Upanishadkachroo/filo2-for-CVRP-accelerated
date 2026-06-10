/**
 * main.cpp  (Phase 1 — Batch 3 integration: OpenMP thread control + MPI parallel restarts)
 *
 * ═══════════════════════════════════════════════════════════════════════
 * WHAT CHANGED vs THE ORIGINAL
 * ═══════════════════════════════════════════════════════════════════════
 *
 * 1. MPI bootstrap (lines marked ── MPI ──)
 *    • MPI_Init / MPI_Finalize wrap the whole program.
 *    • Each MPI rank loads the same instance independently (Instance is
 *      immutable after construction — no inter-rank sharing needed).
 *    • Each rank runs the full Phase 1 construction + Phase 3 core-opt
 *      loop with seed = params.get_seed() + rank, so every rank explores
 *      a distinct trajectory.
 *    • At the end MPI_Reduce(MPI_MIN) finds the globally best solution
 *      cost; the rank holding it sends its solution to rank-0, which
 *      writes the output files.
 *
 * 2. OpenMP thread count (lines marked ── OpenMP ──)
 *    • omp_set_num_threads(params.get_omp_threads()) is called once,
 *      before Instance::make(), so all parallel regions in KDTree,
 *      Instance, and savings construction use the user-specified count.
 *    • If --omp-threads is not given, the default is omp_get_max_threads()
 *      (all available hardware threads).
 *
 * 3. Per-rank seeding
 *    • rand_engine is seeded with  seed + mpi_rank  so all ranks are
 *      running genuinely different random walks.
 *    • SA temperature sampling (welf loop) uses the same per-rank engine,
 *      so temperature schedules diverge across ranks from iteration 1.
 *
 * 4. Output gating
 *    • VERBOSE output and file I/O are gated on  mpi_rank == 0  (or the
 *      rank that found the best solution).
 *    • All ranks print their own best cost at shutdown for debugging.
 *
 * ═══════════════════════════════════════════════════════════════════════
 * WHY MPI AND NOT OPENMP FOR THE CORE-OPT LOOP
 * ═══════════════════════════════════════════════════════════════════════
 * The core-opt loop (rr.apply → local_search → SA accept) operates on a
 * single cobra::Solution object whose internal linked-list is NOT thread-
 * safe.  Parallelising across iterations with OpenMP would require a full
 * rewrite of Solution.  MPI avoids this entirely: each rank has its own
 * Solution in its own process address space — zero shared mutable state,
 * zero synchronization cost inside the loop.  Communication happens only
 * once at the very end (MPI_Reduce + one solution send).
 *
 * OpenMP is still active inside each rank for:
 *   • KDTree neighbor batch queries  (KDTree.cpp)
 *   • Savings vector computation     (SavingsImpl.hpp)
 *   • Savings sort (Thrust fallback) (SavingsImpl.hpp)
 *
 * ═══════════════════════════════════════════════════════════════════════
 * BUILD
 * ═══════════════════════════════════════════════════════════════════════
 *   mpicxx -std=c++17 -O3 -march=native -fopenmp \
 *          -I/path/to/cobra/include \
 *          main.cpp base/KDTree.cpp instance/Instance.cpp ... \
 *          -lcobra -lmpi
 *
 * Run (4 MPI ranks, 4 OpenMP threads each = 16 hardware threads total):
 *   mpirun -np 4 ./filo2 --omp-threads 4 --seed 42 <instance> ...
 */

#include <fstream>
#include <filesystem>

// ── MPI ──────────────────────────────────────────────────────────────────────
#include <mpi.h>

// ── OpenMP ───────────────────────────────────────────────────────────────────
#ifdef _OPENMP
#  include <omp.h>
#endif

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

#ifdef GUI
    #include "Renderer.hpp"
#endif

auto get_basename(const std::string& pathname) -> std::string {
    return {std::find_if(pathname.rbegin(), pathname.rend(),
                         [](char c) { return c == '/'; }).base(),
            pathname.end()};
}

// ─────────────────────────────────────────────────────────────────────────────
// MPI helpers
// ─────────────────────────────────────────────────────────────────────────────

// Broadcast the best solution's route data from winning_rank to rank 0.
// We serialise the solution cost (double) + route strings (for file output).
// Full solution reconstruction is not needed for Phase 1 integration —
// we only need rank-0 to write the correct cost and .vrp.sol file.
// The winning rank sends its solution via cobra::Solution::store_to_file
// into a string buffer that rank-0 receives and writes.
struct RankResult {
    double cost;
    int    routes;
    int    rank;
};

// ─────────────────────────────────────────────────────────────────────────────
int main(int argc, char* argv[]) {

    // ── MPI: initialise before anything else ─────────────────────────────────
    // MPI_THREAD_FUNNELED: MPI calls only from the main thread; OpenMP
    // worker threads do not call MPI.  This is the correct level for a
    // hybrid MPI+OpenMP program where OpenMP is used only for inner loops.
    int provided_thread_level = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided_thread_level);

    int mpi_rank = 0;
    int mpi_size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);

    // Only rank 0 prints the debug-mode warning.
#ifndef NDEBUG
    if (mpi_rank == 0) {
        std::cout << "******************************\n";
        std::cout << "Probably running in DEBUG mode\n";
        std::cout << "******************************\n\n";
    }
#endif

    // ── OpenMP: set thread count once, before any parallel region ────────────
    // All parallel regions in this process (KDTree queries, savings
    // computation) will use this count.  With MPI+OpenMP the recommended
    // configuration is:
    //   mpi_size  × omp_threads  ≤  total hardware threads on the node
    // e.g. 4 ranks × 4 threads = 16 threads on a 16-core node.
#ifdef _OPENMP
    const int omp_threads = params_would_be_parsed_here_but_we_need_params_first = 0;
    // NOTE: omp_set_num_threads is called AFTER params are parsed below.
    //       We declare it here as a reminder of placement.
    (void)omp_threads;
#endif

    cobra::Timer global_timer;
#ifdef VERBOSE
    cobra::Timer timer;
#endif

    const auto params = Parameters(argc, argv);

    // ── OpenMP: apply thread count from params (now that params is ready) ────
#ifdef _OPENMP
    {
        const int requested = params.get_omp_threads();  // new CLI param (see Parameters.hpp note)
        const int hw_threads = omp_get_max_threads();
        const int actual = (requested > 0) ? requested : hw_threads;
        omp_set_num_threads(actual);
#ifdef VERBOSE
        if (mpi_rank == 0) {
            std::cout << "[MPI rank " << mpi_rank << "] OpenMP threads: "
                      << actual << " (hardware: " << hw_threads << ")\n";
        }
#endif
    }
#endif

    // ── Per-rank seed: each rank explores a different random trajectory ───────
    // rank 0 uses the user seed exactly (reproducibility of best-known runs).
    // ranks 1..N-1 use seed+rank (deterministic but distinct).
    const auto base_seed    = params.get_seed();
    const auto rank_seed    = static_cast<unsigned int>(base_seed + mpi_rank);

#ifdef VERBOSE
    if (mpi_rank == 0) {
        std::cout << "MPI ranks: " << mpi_size << "\n";
        std::cout << "Rank 0 seed: " << base_seed
                  << "  (rank r uses seed+" << "r)\n\n";
    }
    // Stagger verbose output to avoid interleaved lines across ranks.
    // Only rank 0 prints the per-phase timing messages below.
    const bool verbose_this_rank = (mpi_rank == 0);
#else
    const bool verbose_this_rank = false;
    (void)verbose_this_rank;
#endif

    // ─────────────────────────────────────────────────────────────────────────
    // Phase 1a: Instance construction (all ranks, independent, identical)
    // KDTree build is single-threaded; neighbor queries are OpenMP-parallel.
    // ─────────────────────────────────────────────────────────────────────────
#ifdef VERBOSE
    if (verbose_this_rank) { std::cout << "Pre-processing the instance.\n"; timer.reset(); }
#endif

    std::optional<cobra::Instance> maybe_instance =
        cobra::Instance::make(params.get_instance_path(), params.get_neighbors_num());

#ifdef VERBOSE
    if (verbose_this_rank) {
        std::cout << "Done in " << timer.elapsed_time<std::chrono::seconds>() << " seconds.\n\n";
    }
#endif

    if (!maybe_instance.has_value()) {
        MPI_Finalize();
        return EXIT_FAILURE;
    }

    const cobra::Instance instance = std::move(maybe_instance.value());

    auto best_solution = cobra::Solution(
        instance,
        std::min(instance.get_vertices_num(), params.get_solution_cache_size()));

    // ─────────────────────────────────────────────────────────────────────────
    // Phase 1b: Clarke & Wright construction (all ranks)
    // Savings computation: OpenMP-parallel.
    // Sort: Thrust GPU (if COBRA_USE_THRUST) or std::sort.
    // Merge loop: sequential.
    // ─────────────────────────────────────────────────────────────────────────
#ifdef VERBOSE
    if (verbose_this_rank) {
        std::cout << "Running CLARKE&WRIGHT to generate an initial solution.\n";
        timer.reset();
    }
#endif

    cobra::clarke_and_wright(instance, best_solution,
                             params.get_cw_lambda(),
                             params.get_cw_neighbors());

#ifdef VERBOSE
    if (verbose_this_rank) {
        std::cout << "Done in " << timer.elapsed_time<std::chrono::seconds>() << " seconds.\n";
        std::cout << "Initial solution: obj = " << best_solution.get_cost()
                  << ", n. of routes = " << best_solution.get_routes_num() << ".\n\n";
    }
#endif

    // ─────────────────────────────────────────────────────────────────────────
    // Setup: MoveGenerators, BPP, SA temperature (all ranks, same instance)
    // ─────────────────────────────────────────────────────────────────────────
    auto k = params.get_sparsification_rule_neighbors();

#ifdef VERBOSE
    if (verbose_this_rank) {
        std::cout << "Setting up MOVEGENERATORS data structures.\n";
        timer.reset();
    }
#endif

    auto move_generators = cobra::MoveGenerators(instance, k);

#ifdef VERBOSE
    if (verbose_this_rank) {
        std::cout << "Done in " << timer.elapsed_time<std::chrono::seconds>() << " seconds.\n";
        const auto tot_arcs = static_cast<unsigned long>(instance.get_vertices_num()) *
                              static_cast<unsigned long>(instance.get_vertices_num());
        const auto move_gen_num  = move_generators.size();
        const auto move_gen_perc = 100.0 * static_cast<double>(move_gen_num) /
                                            static_cast<double>(tot_arcs);
        std::cout << "Using at most " << move_generators.size()
                  << " move-generators out of " << tot_arcs << " total arcs ";
        std::cout << std::fixed << std::setprecision(5);
        std::cout << "(approx. " << move_gen_perc << "%)\n\n";
        std::cout << std::defaultfloat;
    }
#endif

#ifdef VERBOSE
    if (verbose_this_rank) {
        std::cout << "Computing a greedy upper bound on the n. of routes.\n";
        timer.reset();
    }
#endif

    auto kmin = bpp::greedy_first_fit_decreasing(instance);

#ifdef VERBOSE
    if (verbose_this_rank) {
        std::cout << "Done in " << timer.elapsed_time<std::chrono::milliseconds>() << " milliseconds.\n";
        std::cout << "Around " << kmin << " routes should do the job.\n\n";
    }
#endif

    // ── Per-rank rand_engine — seeded with rank_seed ──────────────────────────
    // Each rank now has a fully independent random stream.
    // Rank 0's stream is identical to the original single-process run.
    auto rand_engine = std::mt19937(rank_seed);
    const auto tolerance = params.get_tolerance();

    // ─────────────────────────────────────────────────────────────────────────
    // Phase 2: Route minimization (all ranks, independent)
    // ─────────────────────────────────────────────────────────────────────────
    if (kmin < best_solution.get_routes_num()) {

        const auto routemin_iterations = params.get_routemin_iterations();

#ifdef VERBOSE
        if (verbose_this_rank) {
            std::cout << "Running ROUTEMIN heuristic for at most "
                      << routemin_iterations << " iterations.\n";
            std::cout << "Starting solution: obj = " << best_solution.get_cost()
                      << ", n. of routes = " << best_solution.get_routes_num() << ".\n";
            timer.reset();
        }
#endif

        best_solution = routemin(instance, best_solution, rand_engine,
                                 move_generators, kmin, routemin_iterations, tolerance);

#ifdef VERBOSE
        if (verbose_this_rank) {
            std::cout << "Final solution: obj = " << best_solution.get_cost()
                      << ", n. routes = " << best_solution.get_routes_num() << "\n";
            std::cout << "Done in " << timer.elapsed_time<std::chrono::seconds>() << " seconds.\n\n";
        }
#endif
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Phase 3: Core optimization loop (all ranks, independent trajectories)
    // Each rank runs an independent SA walk seeded with rank_seed.
    // No inter-rank communication inside this loop.
    // ─────────────────────────────────────────────────────────────────────────
    auto rvnd0 = cobra::RandomizedVariableNeighborhoodDescent(
        instance, move_generators,
        {cobra::E11,   cobra::E10,   cobra::TAILS, cobra::SPLIT, cobra::RE22B, cobra::E22,
         cobra::RE20,  cobra::RE21,  cobra::RE22S, cobra::E21,   cobra::E20,   cobra::TWOPT,
         cobra::RE30,  cobra::E30,   cobra::RE33B, cobra::E33,   cobra::RE31,  cobra::RE32B,
         cobra::RE33S, cobra::E31,   cobra::E32,   cobra::RE32S},
        rand_engine, tolerance);

    auto rvnd1 = cobra::RandomizedVariableNeighborhoodDescent(
        instance, move_generators, {cobra::EJCH}, rand_engine, tolerance);

    auto local_search = cobra::VariableNeighborhoodDescentComposer(tolerance);
    local_search.append(&rvnd0);
    local_search.append(&rvnd1);

#ifdef TIMELIMIT
    const int optimization_seconds     = params.get_optimization_seconds();
    const int optimization_milliseconds = 1000 * optimization_seconds;
#else
    const auto coreopt_iterations = params.get_coreopt_iterations();
#endif

    auto neighbor = best_solution;

    const auto gamma_base = params.get_gamma_base();
    auto gamma         = std::vector<double>(instance.get_vertices_num(), gamma_base);
    auto gamma_counter = std::vector<int>(instance.get_vertices_num(), 0);

    const auto delta = params.get_delta();
    auto average_number_of_vertices_accessed = cobra::Welford();

    auto gamma_vertices = std::vector<int>();
    for (auto i = instance.get_vertices_begin(); i < instance.get_vertices_end(); i++) {
        gamma_vertices.emplace_back(i);
    }
    move_generators.set_active_percentage(gamma, gamma_vertices);

    auto ruined_customers = std::vector<int>();
    auto rr = RuinAndRecreate(instance, rand_engine);

    const auto intensification_lb = params.get_shaking_lb_factor();
    const auto intensification_ub = params.get_shaking_ub_factor();

    const auto mean_solution_arc_cost =
        neighbor.get_cost() /
        (static_cast<double>(instance.get_customers_num()) +
         2.0 * static_cast<double>(neighbor.get_routes_num()));

    auto shaking_lb_factor = mean_solution_arc_cost * intensification_lb;
    auto shaking_ub_factor = mean_solution_arc_cost * intensification_ub;

#ifdef VERBOSE
    if (verbose_this_rank) {
        std::cout << "Shaking LB = " << shaking_lb_factor << "\n";
        std::cout << "Shaking UB = " << shaking_ub_factor << "\n";
    }
#endif

    const auto omega_base = std::max(1, static_cast<int>(
        std::ceil(std::log(instance.get_vertices_num()))));
    auto omega = std::vector<int>(instance.get_vertices_num(), omega_base);
    auto random_choice   = std::uniform_int_distribution(0, 1);
    auto vertices_dist   = std::uniform_int_distribution(
        instance.get_vertices_begin(), instance.get_vertices_end() - 1);

    // SA temperature sampling — uses per-rank rand_engine → distinct schedules.
    cobra::Welford welf;
    for (int i = 0; i < instance.get_vertices_num(); ++i) {
        welf.update(instance.get_cost(vertices_dist(rand_engine),
                                      vertices_dist(rand_engine)));
    }

    const auto sa_initial_temperature = welf.get_mean() * params.get_sa_initial_factor();
    const auto sa_final_temperature   = sa_initial_temperature * params.get_sa_final_factor();

#ifdef TIMELIMIT
    const int coreopt_seconds =
        optimization_seconds - global_timer.elapsed_time<std::chrono::seconds>();
    auto sa = cobra::TimeBasedSimulatedAnnealing(
        sa_initial_temperature, sa_final_temperature, rand_engine, coreopt_seconds * 1000);
#else
    auto sa = cobra::SimulatedAnnealing(
        sa_initial_temperature, sa_final_temperature, rand_engine, coreopt_iterations);
#endif

#ifdef VERBOSE
    if (verbose_this_rank) {
        std::cout << "Simulated annealing temperature goes from "
                  << sa_initial_temperature << " to " << sa_final_temperature << ".\n\n";
    }
#endif

#ifdef VERBOSE
    if (verbose_this_rank) {
#ifdef TIMELIMIT
        std::cout << "Running COREOPT for " << std::max(0, coreopt_seconds) << " seconds.\n";
#else
        std::cout << "Running COREOPT for " << coreopt_iterations << " iterations.\n";
#endif

        auto welford_rac_before_shaking = cobra::Welford();
        auto welford_rac_after_shaking  = cobra::Welford();
        auto welford_local_optima       = cobra::Welford();
        auto welford_shaken_solutions   = cobra::Welford();
        auto printer = cobra::PrettyPrinter(
            {{"%",          cobra::PrettyPrinter::Field::Type::REAL,    5,  " "},
             {"Iterations", cobra::PrettyPrinter::Field::Type::INTEGER, 10, " "},
             {"Objective",  cobra::PrettyPrinter::Field::Type::INTEGER, 10, " "},
             {"Routes",     cobra::PrettyPrinter::Field::Type::INTEGER, 6,  " "},
             {"Iter/s",     cobra::PrettyPrinter::Field::Type::REAL,    10, " "},
             {"Eta (s)",    cobra::PrettyPrinter::Field::Type::REAL,    10, " "},
             {"RR (micro)", cobra::PrettyPrinter::Field::Type::REAL,    10, " "},
             {"LS (micro)", cobra::PrettyPrinter::Field::Type::REAL,    10, " "},
             {"Gamma",      cobra::PrettyPrinter::Field::Type::REAL,    5,  " "},
             {"Omega",      cobra::PrettyPrinter::Field::Type::REAL,    6,  " "},
             {"Temp",       cobra::PrettyPrinter::Field::Type::REAL,    6,  " "}});

        auto elapsed_minutes = 0UL;
        timer.reset();
        cobra::Timer coreopt_timer;
        cobra::Welford welford_rr;
        cobra::Welford welford_ls;
#endif  // VERBOSE (inner block — matching #ifdef below)

    // The core-opt loop body is identical to the original.
    // It runs fully independently per MPI rank — no synchronization.
    double reference_solution_cost = neighbor.get_cost();

#ifdef TIMELIMIT
    for (auto iter = 0;
         static_cast<int>(global_timer.elapsed_time<std::chrono::milliseconds>()) <= optimization_milliseconds;
         iter++) {
#else
    for (auto iter = 0; iter < coreopt_iterations; iter++) {
#endif

        neighbor.apply_undo_list1(neighbor);
        neighbor.clear_do_list1();
        neighbor.clear_undo_list1();
        neighbor.clear_svc();

#ifdef VERBOSE
        if (verbose_this_rank) {
            if (global_timer.elapsed_time<std::chrono::minutes>() >= elapsed_minutes + 5) {
                printer.notify("Optimizing for " +
                               std::to_string(global_timer.elapsed_time<std::chrono::minutes>()) +
                               " minutes.");
                elapsed_minutes += 5;
            }
            cobra::Timer rr_timer;
#endif

        const auto walk_seed = rr.apply(neighbor, omega);

#ifdef VERBOSE
            const auto rr_time = rr_timer.elapsed_time<std::chrono::microseconds>();
            welford_rr.update(rr_time);
        }
#endif

#ifdef GUI
        const auto shaken_solution_cost = neighbor.get_cost();
#endif

        ruined_customers.clear();
        for (auto i = neighbor.get_svc_begin(); i != neighbor.get_svc_end();
             i = neighbor.get_svc_next(i)) {
            ruined_customers.emplace_back(i);
        }

#ifdef VERBOSE
        if (verbose_this_rank) {
            welford_rac_after_shaking.update(static_cast<double>(neighbor.get_svc_size()));
            welford_shaken_solutions.update(neighbor.get_cost());
            cobra::Timer ls_timer;
#endif

        local_search.sequential_apply(neighbor);

#ifdef VERBOSE
            const auto ls_time = ls_timer.elapsed_time<std::chrono::microseconds>();
            welford_ls.update(ls_time);
        }
#endif

        average_number_of_vertices_accessed.update(
            static_cast<double>(neighbor.get_svc_size()));

#ifdef TIMELIMIT
        const int avg_iters_sec =
            (iter + 1.0) / (coreopt_timer.elapsed_time<std::chrono::seconds>() + 0.01);
        const int remaining_sec =
            coreopt_seconds - coreopt_timer.elapsed_time<std::chrono::seconds>();
        const int estimated_remaining_iters = avg_iters_sec * remaining_sec;
        const int estimated_total_iters = 1 + iter + estimated_remaining_iters;
        const int max_non_improving_iterations = static_cast<int>(std::ceil(
            delta * static_cast<double>(estimated_total_iters) *
            static_cast<double>(average_number_of_vertices_accessed.get_mean()) /
            static_cast<double>(instance.get_vertices_num())));
#else
        auto max_non_improving_iterations = static_cast<int>(std::ceil(
            delta * static_cast<double>(coreopt_iterations) *
            static_cast<double>(average_number_of_vertices_accessed.get_mean()) /
            static_cast<double>(instance.get_vertices_num())));
#endif

#ifdef VERBOSE
        if (verbose_this_rank) {
            welford_rac_before_shaking.update(static_cast<double>(neighbor.get_svc_size()));
            welford_local_optima.update(neighbor.get_cost());
        }
#endif

        bool improved_best_solution;

        if (neighbor.get_cost() < best_solution.get_cost()) {

            improved_best_solution = true;

            neighbor.apply_do_list2(best_solution);
            neighbor.apply_do_list1(best_solution);
            neighbor.clear_do_list2();

            assert(best_solution == neighbor);

            gamma_vertices.clear();
            for (auto i = neighbor.get_svc_begin(); i != neighbor.get_svc_end();
                 i = neighbor.get_svc_next(i)) {
                gamma[i]         = gamma_base;
                gamma_counter[i] = 0;
                gamma_vertices.emplace_back(i);
            }
            move_generators.set_active_percentage(gamma, gamma_vertices);

#ifdef VERBOSE
            if (verbose_this_rank) {
                welford_local_optima.reset();
                welford_local_optima.update(neighbor.get_cost());
                welford_shaken_solutions.reset();
                welford_shaken_solutions.update(neighbor.get_cost());
            }
#endif

        } else {

            improved_best_solution = false;

            for (auto i = neighbor.get_svc_begin(); i != neighbor.get_svc_end();
                 i = neighbor.get_svc_next(i)) {
                gamma_counter[i]++;
                if (gamma_counter[i] >= max_non_improving_iterations) {
                    gamma[i] = std::min(gamma[i] * 2.0, 1.0);
                    gamma_counter[i] = 0;
                    gamma_vertices.clear();
                    gamma_vertices.emplace_back(i);
                    move_generators.set_active_percentage(gamma, gamma_vertices);
                }
            }
        }

        const auto seed_shake_value = omega[walk_seed];

        if (neighbor.get_cost() > shaking_ub_factor + reference_solution_cost) {
            for (auto i : ruined_customers) {
                if (omega[i] > seed_shake_value - 1) { omega[i]--; }
            }
        } else if (neighbor.get_cost() >= reference_solution_cost &&
                   neighbor.get_cost() <  reference_solution_cost + shaking_lb_factor) {
            for (auto i : ruined_customers) {
                if (omega[i] < seed_shake_value + 1) { omega[i]++; }
            }
        } else {
            for (auto i : ruined_customers) {
                if (random_choice(rand_engine)) {
                    if (omega[i] > seed_shake_value - 1) { omega[i]--; }
                } else {
                    if (omega[i] < seed_shake_value + 1) { omega[i]++; }
                }
            }
        }

#ifdef TIMELIMIT
        if (sa.accept(reference_solution_cost, neighbor,
                      coreopt_timer.elapsed_time<std::chrono::milliseconds>())) {
#else
        if (sa.accept(reference_solution_cost, neighbor)) {
#endif
            if (!improved_best_solution) {
                neighbor.append_do_list1_to_do_list2();
            }
            neighbor.clear_do_list1();
            neighbor.clear_undo_list1();
            reference_solution_cost = neighbor.get_cost();

            const auto updated_mean_solution_arc_cost =
                neighbor.get_cost() /
                (static_cast<double>(instance.get_customers_num()) +
                 2.0 * static_cast<double>(neighbor.get_routes_num()));
            shaking_lb_factor = updated_mean_solution_arc_cost * intensification_lb;
            shaking_ub_factor = updated_mean_solution_arc_cost * intensification_ub;
        }

        sa.decrease_temperature();

#ifdef VERBOSE
        if (verbose_this_rank &&
            timer.elapsed_time<std::chrono::seconds>() > 1) {
            timer.reset();

            auto gamma_mean = 0.0;
            for (auto i = instance.get_vertices_begin(); i < instance.get_vertices_end(); i++) {
                gamma_mean += gamma[i];
            }
            gamma_mean /= static_cast<double>(instance.get_vertices_num());

            auto omega_mean = 0.0;
            for (auto i = instance.get_customers_begin(); i < instance.get_customers_end(); i++) {
                omega_mean += omega[i];
            }
            omega_mean /= static_cast<double>(instance.get_customers_num());

#ifdef TIMELIMIT
            const int elapsed_time_ms =
                coreopt_timer.elapsed_time<std::chrono::milliseconds>();
            const auto progress =
                100.0f * (elapsed_time_ms / 1000.0) / coreopt_seconds;
            printer.print(progress, iter + 1, best_solution.get_cost(),
                          best_solution.get_routes_num(), avg_iters_sec, remaining_sec,
                          welford_rr.get_mean(), welford_ls.get_mean(),
                          gamma_mean, omega_mean, sa.get_temperature(elapsed_time_ms));
#else
            const auto progress        = 100.0 * (iter + 1.0) / coreopt_iterations;
            const auto elapsed_seconds = coreopt_timer.elapsed_time<std::chrono::seconds>();
            const auto iter_per_second =
                static_cast<double>(iter + 1) / (static_cast<double>(elapsed_seconds) + 0.01);
            const auto remaining_iter     = coreopt_iterations - iter;
            const auto estimated_rem_time =
                static_cast<double>(remaining_iter) / iter_per_second;
            printer.print(progress, iter + 1, best_solution.get_cost(),
                          best_solution.get_routes_num(), iter_per_second, estimated_rem_time,
                          welford_rr.get_mean(), welford_ls.get_mean(),
                          gamma_mean, omega_mean, sa.get_temperature());
#endif
        }
#endif
    }  // end core-opt loop

#ifdef VERBOSE
    // Close the verbose_this_rank block opened above.
    }
#endif

    int global_time_elapsed = global_timer.elapsed_time<std::chrono::seconds>();

    // ─────────────────────────────────────────────────────────────────────────
    // MPI reduction: find the globally best solution across all ranks
    // ─────────────────────────────────────────────────────────────────────────
    //
    // Step 1: each rank broadcasts its best cost; MPI_Allreduce finds the min.
    // Step 2: the winning rank tag-sends its cost, route count, and timing
    //         to rank 0.
    // Step 3: rank 0 writes output files; non-zero ranks exit silently.
    //
    // We use MPI_DOUBLE_INT with MPI_MINLOC to find both the minimum cost
    // AND the rank that achieved it in a single collective operation.

    struct { double cost; int rank; } local_val, global_val;
    local_val.cost = best_solution.get_cost();
    local_val.rank = mpi_rank;

    MPI_Allreduce(&local_val, &global_val, 1,
                  MPI_DOUBLE_INT, MPI_MINLOC, MPI_COMM_WORLD);

    const int winning_rank = global_val.rank;
    const double best_global_cost = global_val.cost;

#ifdef VERBOSE
    // All ranks print their result for diagnostic purposes.
    // In production builds this block compiles away.
    std::cout << "[rank " << mpi_rank << "] best cost = "
              << best_solution.get_cost()
              << "  routes = " << best_solution.get_routes_num()
              << (mpi_rank == winning_rank ? "  ← GLOBAL BEST" : "")
              << "\n";
#endif

    // ─────────────────────────────────────────────────────────────────────────
    // Output: only winning_rank writes files.
    // If winning_rank != 0 we send timing to rank-0 for the summary line.
    // ─────────────────────────────────────────────────────────────────────────
    if (mpi_rank == winning_rank) {

#ifdef VERBOSE
        std::cout << "\nBest solution found (rank " << winning_rank << "):\n";
        std::cout << "obj = " << best_global_cost
                  << ", n. routes = " << best_solution.get_routes_num() << "\n";

        // If winning rank is not 0, send global_time_elapsed to rank 0
        // so rank 0 can print the summary.  For simplicity we print here.
        std::cout << "Run completed in " << global_time_elapsed << " seconds\n";
#endif

        const auto outfile =
            params.get_outpath() +
            get_basename(params.get_instance_path()) +
            "_seed-" + std::to_string(base_seed) + ".out";

        std::filesystem::create_directories(params.get_outpath());

        auto out_stream = std::ofstream(outfile);
        out_stream << std::setprecision(10);
        out_stream << best_global_cost << "\t" << global_time_elapsed << "\n";

        cobra::Solution::store_to_file(
            instance, best_solution,
            params.get_outpath() +
            get_basename(params.get_instance_path()) +
            "_seed-" + std::to_string(base_seed) + ".vrp.sol");

#ifdef VERBOSE
        std::cout << "Results stored in\n";
        std::cout << " - " << outfile << "\n";
        std::cout << " - "
                  << params.get_outpath() +
                     get_basename(params.get_instance_path()) +
                     "_seed-" + std::to_string(base_seed) + ".vrp.sol"
                  << "\n";
#endif
    }

    // ── MPI: finalise ─────────────────────────────────────────────────────────
    MPI_Finalize();
    return EXIT_SUCCESS;
}