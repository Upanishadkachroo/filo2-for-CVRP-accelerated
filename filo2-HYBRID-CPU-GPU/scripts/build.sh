#!/usr/bin/env bash
# scripts/build.sh
# Build FILO2-CUDA hybrid solver.
#
# Environment variables (all optional)
#   GPU_ARCH   – nvcc -arch flag  (default: autodetect via nvidia-smi)
#   CPU_ONLY   – set to 1 to skip all CUDA sources and build baseline
#   DEBUG      – set to 1 for debug build (-G -g -O0)
#   CXX        – host C++ compiler  (default: g++)
#   OUT        – output binary name (default: filo2_cuda)

set -euo pipefail

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
CXX="${CXX:-g++}"
OUT="${OUT:-filo2_cuda}"
CPU_ONLY="${CPU_ONLY:-0}"
DEBUG="${DEBUG:-0}"

# Auto-detect GPU architecture if not provided
if [[ -z "${GPU_ARCH:-}" ]]; then
    if command -v nvidia-smi &>/dev/null; then
        SM=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader \
             | head -1 | tr -d '.')
        GPU_ARCH="sm_${SM}"
    else
        GPU_ARCH="sm_75"   # safe fallback: Turing
        echo "[build] nvidia-smi not found, defaulting to -arch=${GPU_ARCH}"
    fi
fi

echo "[build] Target architecture: ${GPU_ARCH}"

# ---------------------------------------------------------------------------
# Source lists
# ---------------------------------------------------------------------------
CPP_SOURCES=(
    main.cpp
)

CUDA_SOURCES=(
    instance/Instance.cu
    solution/savings_gpu.cu
    movegen/knn_gpu.cu
    localsearch/delta_gpu.cu
    localsearch/feasibility_gpu.cu
    opt/recreate_gpu.cu
)

# ---------------------------------------------------------------------------
# Compiler flags
# ---------------------------------------------------------------------------
if [[ "${DEBUG}" == "1" ]]; then
    OPT_FLAGS="-G -g -O0"
    HOST_OPT="-O0 -g"
    echo "[build] Debug mode"
else
    OPT_FLAGS="-O3 --use_fast_math"
    HOST_OPT="-O3 -march=native"
fi

NVCC_FLAGS=(
    -std=c++17
    -arch="${GPU_ARCH}"
    ${OPT_FLAGS}
    -Xcompiler "${HOST_OPT}"
    -Xcompiler "-Wall"
    --expt-relaxed-constexpr
)

# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------
if [[ "${CPU_ONLY}" == "1" ]]; then
    echo "[build] CPU-only mode — skipping CUDA sources"
    "${CXX}" -std=c++17 ${HOST_OPT} -DCPU_ONLY \
        "${CPP_SOURCES[@]}" \
        -o "${OUT}"
else
    # Compile all sources (nvcc acts as both CUDA and host C++ compiler)
    nvcc "${NVCC_FLAGS[@]}" \
        "${CPP_SOURCES[@]}" \
        "${CUDA_SOURCES[@]}" \
        -o "${OUT}"
fi

echo "[build] Done → ./${OUT}"
