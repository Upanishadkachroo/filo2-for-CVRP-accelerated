#ifndef FILO2_CUDA_SAVINGS_KERNEL_HPP
#define FILO2_CUDA_SAVINGS_KERNEL_HPP

#include <vector>
#include "../instance/Instance.hpp"

namespace cobra {

struct SavingGPU {
    float key;      // negated savings value (for ascending sort)
    uint64_t value; // packed (i << 32) | j
};

// Main GPU entry point: computes and sorts all savings, returns vector sorted descending by savings.
bool computeSavingsGPU(const Instance& instance, int neighbors_num,
                       std::vector<Saving>& savings);

} // namespace cobra

#endif
