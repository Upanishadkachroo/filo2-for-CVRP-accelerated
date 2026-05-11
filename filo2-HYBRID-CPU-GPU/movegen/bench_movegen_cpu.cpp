// bench_movegen_cpu.cpp — benchmarks MoveGenerators CPU build time
// Compile (from instance/ directory):
//   g++ -O3 -march=native -std=c++17 \
//       -I. -I.. -I../base -I../movegen \
//       -I/home/upanishad/Desktop/filo2-for-CVRP-accelerated/filo2-main/instance \
//       -I/home/upanishad/Desktop/filo2-for-CVRP-accelerated/filo2-main/base \
//       -I/home/upanishad/Desktop/filo2-for-CVRP-accelerated/filo2-main/movegen \
//       bench_movegen_cpu.cpp \
//       /home/upanishad/Desktop/filo2-for-CVRP-accelerated/filo2-main/instance/Instance.cpp \
//       /home/upanishad/Desktop/filo2-for-CVRP-accelerated/filo2-main/instance/Parser.cpp \
//       /home/upanishad/Desktop/filo2-for-CVRP-accelerated/filo2-main/base/KDTree.cpp \
//       -o bench_movegen_cpu

#include "Instance.hpp"

// Pull in MoveGenerators from filo2-main (header-only apart from Instance)
#include "/home/upanishad/Desktop/filo2-for-CVRP-accelerated/filo2-main/movegen/MoveGenerators.hpp"

#include <chrono>
#include <fstream>
#include <iostream>
#include <string>

using namespace std::chrono;

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "usage: bench_movegen_cpu <file.vrp> [k]\n";
        return 1;
    }
    std::string path = argv[1];
    int k = argc > 2 ? std::stoi(argv[2]) : 10;

    // Load instance
    auto inst_opt = cobra::Instance::make(path, k + 1);
    if (!inst_opt) { std::cerr << "load failed: " << path << "\n"; return 1; }
    auto& inst = *inst_opt;
    const int N = inst.get_vertices_num();

    // --- Benchmark: how long does MoveGenerators construction take? ---
    // This measures build_base_moves() which iterates k-NN pairs and
    // inserts them into the move list — the CPU equivalent of knn_kernel.

    static constexpr double TARGET_SEC = 5.0;
    int reps = 0;
    double total_ms = 0.0;
    size_t total_moves = 0;

    auto wall0 = high_resolution_clock::now();

    while (true) {
        auto t0 = high_resolution_clock::now();
        cobra::MoveGenerators mg(inst, k);
        auto t1 = high_resolution_clock::now();

        total_ms  += duration<double, std::milli>(t1 - t0).count();
        total_moves = mg.size();   // capture once (same every rep)
        reps++;

        double elapsed = duration<double>(high_resolution_clock::now() - wall0).count();
        if (elapsed >= TARGET_SEC) break;
    }

    double avg_ms        = total_ms / reps;
    double moves_per_sec = (total_moves * reps) / (total_ms / 1000.0);

    std::cout << "instance="   << path
              << "  n="        << N
              << "  k="        << k
              << "  moves="    << total_moves
              << "  avg_build_ms=" << avg_ms
              << "  Mmoves/s=" << moves_per_sec / 1e6
              << "\n";

    // --- Also benchmark get_cost() for direct comparison ---
    static constexpr double COST_SEC = 2.0;
    long long cost_calls = 0;
    auto c0 = high_resolution_clock::now();
    volatile double sink = 0;
    while (true) {
        for (int i = 0; i < N; ++i)
            sink += inst.get_cost(i, (i+1) % N);
        cost_calls += N;
        double elapsed = duration<double>(high_resolution_clock::now() - c0).count();
        if (elapsed >= COST_SEC) break;
    }
    auto c1 = high_resolution_clock::now();
    double cost_ms     = duration<double, std::milli>(c1 - c0).count();
    double ns_per_call = cost_ms * 1e6 / cost_calls;
    (void)sink;

    // --- CSV output ---
    bool write_header = false;
    { std::ifstream chk("movegen_results.csv"); write_header = !chk.good(); }

    std::ofstream csv("movegen_results.csv", std::ios::app);
    if (write_header)
        csv << "instance,n_vertices,k,total_moves,avg_build_ms,"
               "Mmoves_per_sec,cost_ns_per_call\n";

    csv << path << ","
        << N << ","
        << k << ","
        << total_moves << ","
        << avg_ms << ","
        << moves_per_sec / 1e6 << ","
        << ns_per_call << "\n";

    return 0;
}