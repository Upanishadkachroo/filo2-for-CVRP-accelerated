// bench_cpu.cpp — time-limited benchmark (runs for TARGET_SEC per instance)
// Compile:
//   g++ -O3 -march=native -std=c++17 \
//       -I. -I.. -I../base \
//       -I/home/upanishad/Desktop/filo2-for-CVRP-accelerated/filo2-main/instance \
//       -I/home/upanishad/Desktop/filo2-for-CVRP-accelerated/filo2-main/base \
//       bench_cpu.cpp \
//       /home/upanishad/Desktop/filo2-for-CVRP-accelerated/filo2-main/instance/Instance.cpp \
//       /home/upanishad/Desktop/filo2-for-CVRP-accelerated/filo2-main/instance/Parser.cpp \
//       /home/upanishad/Desktop/filo2-for-CVRP-accelerated/filo2-main/base/KDTree.cpp \
//       -o bench_cpu

#include "Instance.hpp"
#include <chrono>
#include <fstream>
#include <iostream>
#include <string>

using namespace std::chrono;

static constexpr double TARGET_SEC = 2.0;   // run each instance for 2 seconds max

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "usage: bench_cpu <file.vrp> [num_neighbors]\n";
        return 1;
    }
    std::string path = argv[1];
    int k = argc > 2 ? std::stoi(argv[2]) : 10;

    auto inst_opt = cobra::Instance::make(path, k);
    if (!inst_opt) {
        std::cerr << "ERROR: failed to load " << path << "\n";
        return 1;
    }
    auto& inst = *inst_opt;
    const int N = inst.get_vertices_num();

    // --- warm up (one pass, not timed) ---
    volatile double sink = 0;
    for (int i = 0; i < N; ++i)
        sink += inst.get_cost(i, (i + 1) % N);

    // --- timed loop: keep going until TARGET_SEC elapsed ---
    long long total_calls = 0;
    auto t0 = high_resolution_clock::now();

    while (true) {
        for (int i = 0; i < N; ++i)
            sink += inst.get_cost(i, (i + 1) % N);
        total_calls += N;

        double elapsed = duration<double>(high_resolution_clock::now() - t0).count();
        if (elapsed >= TARGET_SEC) break;
    }

    auto t1 = high_resolution_clock::now();
    double total_ms    = duration<double, std::milli>(t1 - t0).count();
    double ns_per_call = total_ms * 1e6 / total_calls;
    double calls_per_sec = total_calls / (total_ms / 1000.0);

    // --- stdout summary ---
    std::cout << "instance=" << path
              << "  n=" << N
              << "  ns/call=" << ns_per_call
              << "  Mcalls/s=" << calls_per_sec / 1e6
              << "\n";

    // --- append CSV row ---
    bool write_header = false;
    {
        std::ifstream check("results.csv");
        write_header = !check.good();
    }
    std::ofstream csv("results.csv", std::ios::app);
    if (write_header)
        csv << "instance,n_vertices,total_ms,total_calls,ns_per_call,Mcalls_per_sec\n";

    csv << path << ","
        << N << ","
        << total_ms << ","
        << total_calls << ","
        << ns_per_call << ","
        << calls_per_sec / 1e6 << "\n";

    (void)sink;
    return 0;
}