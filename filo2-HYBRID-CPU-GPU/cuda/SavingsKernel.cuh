#ifndef FILO2_CUDA_SAVINGS_KERNEL_HPP
#define FILO2_CUDA_SAVINGS_KERNEL_HPP

#include <vector>
#include "../instance/Instance.hpp"
#include "../instance/Saving.hpp"

namespace cobra {

// Compute all savings and sort descending using GPU.
// Returns true on success; output vector is sorted by value descending.
bool computeSavingsGPU(const Instance& instance,
                       int neighbors_num,
                       double lambda,
                       std::vector<Saving>& savings);

} // namespace cobra

#endif
