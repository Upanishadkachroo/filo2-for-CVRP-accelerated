cmake_minimum_required(VERSION 3.22)

# CUDA language must be declared in project() or via enable_language() BEFORE
# any find_package calls that depend on it. We conditionally enable it below.
# First declare the project with just CXX, then add CUDA if requested.
project(filo2 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ---------------------------------------------------------------------------
# Compiler options
# ---------------------------------------------------------------------------
set(WARNING_FLAGS "-Wall -Wextra -Wpedantic -Wuninitialized")
set(OPT_FLAGS "-O3 -march=native -flto=auto")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${WARNING_FLAGS}")
set(CMAKE_CXX_FLAGS_DEBUG
    "${CMAKE_CXX_FLAGS_DEBUG} -fsanitize=address -fsanitize=undefined \
     -fsanitize-address-use-after-scope -fno-omit-frame-pointer -g")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} ${OPT_FLAGS}")

set(CMAKE_VERBOSE_MAKEFILE ON)

# ---------------------------------------------------------------------------
# Build options
# ---------------------------------------------------------------------------
option(ENABLE_VERBOSE    "Enable verbose output"                          ON)
option(ENABLE_GUI        "Enable graphical interface"                     OFF)
option(ENABLE_TIMELIMIT  "Enable time limit termination"                  OFF)
option(ENABLE_CUDA       "Enable CUDA support"                            OFF)
option(USE_GRID_NEIGHBOR "Use grid-based CUDA neighbor finder (large N)"  OFF)

message("-- Build options")

if(ENABLE_VERBOSE)
    message("--- Verbose output ENABLED")
    add_definitions(-DVERBOSE)
else()
    message("--- Verbose output DISABLED")
endif()

if(ENABLE_GUI)
    message("--- Graphical interface ENABLED")
    add_definitions(-DGUI)
    find_package(glfw3 REQUIRED)
    set(LIBRARIES ${LIBRARIES} glfw GL)
else()
    message("--- Graphical interface DISABLED")
endif()

if(ENABLE_TIMELIMIT)
    message("--- Time limit termination ENABLED")
    add_definitions(-DTIMELIMIT)
else()
    message("--- Time limit termination DISABLED")
endif()

# ---------------------------------------------------------------------------
# OpenMP (mandatory for parallel preprocessing)
# ---------------------------------------------------------------------------
find_package(OpenMP REQUIRED)

# ---------------------------------------------------------------------------
# CUDA configuration — modern CMake approach (no deprecated FindCUDA)
# ---------------------------------------------------------------------------
if(ENABLE_CUDA)
    # enable_language(CUDA) must be called before any CUDA-related variables
    # are set; it activates nvcc as the compiler for .cu translation units.
    enable_language(CUDA)

    # CUDAToolkit replaces the old find_package(CUDA) / FindCUDA module.
    find_package(CUDAToolkit REQUIRED)

    message("--- CUDA ENABLED (toolkit: ${CUDAToolkit_VERSION})")
    add_definitions(-DCUDA_ENABLED -DUSE_CUDA_NEIGHBORS)

    # CUDA language standard
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)

    # Target GPU architecture — Colab Tesla T4 = sm_75.
    # Adjust if you run on a different GPU:
    #   A100  -> 80,  V100 -> 70,  RTX 30xx -> 86,  RTX 40xx -> 89
    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
        set(CMAKE_CUDA_ARCHITECTURES 75)
    endif()

    # nvcc optimisation flags (host-side -O3 comes from CXX flags above)
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 --use_fast_math")

    # Make sure nvcc uses the same host compiler as CMake's CXX compiler
    set(CMAKE_CUDA_HOST_COMPILER "${CMAKE_CXX_COMPILER}")
else()
    message("--- CUDA DISABLED")
endif()

# ---------------------------------------------------------------------------
# Source files
# ---------------------------------------------------------------------------
set(SOURCES
    main.cpp
    Parameters.hpp
    Renderer.hpp

    base/KDTree.cpp
    base/KDTree.hpp
    base/Timer.hpp
    base/Welford.hpp
    base/PrettyPrinter.hpp
    base/NonCopyable.hpp
    base/Flat2DVector.hpp
    base/BinaryHeap.hpp
    base/BitMatrix.hpp
    base/FixedSizeValueStack.hpp
    base/functor.hpp
    base/LRUCache.hpp
    base/macro.hpp
    base/SmallFlatMap.hpp
    base/SmallFlatSet.hpp
    base/SparseIntSet.hpp
    base/VectorView.hpp

    instance/Instance.cpp
    instance/Instance.hpp
    instance/Parser.cpp
    instance/Parser.hpp

    localsearch/AbstractOperator.hpp
    localsearch/EjectionChain.hpp
    localsearch/LocalSearch.hpp
    localsearch/OneOneExchange.hpp
    localsearch/OneZeroExchange.hpp
    localsearch/RevThreeOneExchange.hpp
    localsearch/RevThreeThreeExchange.hpp
    localsearch/RevThreeTwoExchange.hpp
    localsearch/RevThreeZeroExchange.hpp
    localsearch/RevTwoOneExchange.hpp
    localsearch/RevTwoTwoExchange.hpp
    localsearch/RevTwoZeroExchange.hpp
    localsearch/SplitExchange.hpp
    localsearch/TailsExchange.hpp
    localsearch/ThreeOneExchange.hpp
    localsearch/ThreeThreeExchange.hpp
    localsearch/ThreeTwoExchange.hpp
    localsearch/ThreeZeroExchange.hpp
    localsearch/TwoOneExchange.hpp
    localsearch/TwoOptExchange.hpp
    localsearch/TwoTwoExchange.hpp
    localsearch/TwoZeroExchange.hpp

    movegen/MoveGenerators.hpp

    opt/routemin.hpp
    opt/RuinAndRecreate.hpp
    opt/SimulatedAnnealing.hpp
    opt/bpp.hpp

    solution/savings.hpp
    solution/Solution.hpp
    solution/Solution.cpp
)

# CUDA-specific sources appended only when CUDA is enabled
if(ENABLE_CUDA)
    list(APPEND SOURCES
        cuda/CudaNeighborFinder.cu
        cuda/CudaNeighborFinder.hpp
    )
    if(USE_GRID_NEIGHBOR)
        list(APPEND SOURCES
            cuda/GridNeighborFinder.cu
            cuda/GridNeighborFinder.hpp
        )
        add_definitions(-DUSE_GRID_NEIGHBOR)
    endif()
endif()

# ---------------------------------------------------------------------------
# Executable target
# ---------------------------------------------------------------------------
add_executable(filo2 ${SOURCES})
set_target_properties(filo2 PROPERTIES OUTPUT_NAME "filo2-HYBRID-CPU-GPU")

# ---------------------------------------------------------------------------
# Link libraries
# ---------------------------------------------------------------------------
target_link_libraries(filo2 PUBLIC ${LIBRARIES} OpenMP::OpenMP_CXX)

if(ENABLE_CUDA)
    # CUDA::cudart is provided by find_package(CUDAToolkit); no manual path needed.
    target_link_libraries(filo2 PUBLIC CUDA::cudart)

    # Expose CUDA headers to all translation units (needed when .cpp files
    # include .cuh or cuda_runtime.h directly).
    target_include_directories(filo2 PRIVATE ${CUDAToolkit_INCLUDE_DIRS})

    # Tell nvcc to compile device code as relocatable so it can be linked with
    # host objects produced by g++.
    set_target_properties(filo2 PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
endif()

# OpenMP compile flag fallback (belt-and-suspenders for older CMake generators)
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    target_compile_options(filo2 PRIVATE -fopenmp)
endif()