cmake_minimum_required(VERSION 3.22)

# Project declared with CXX only; CUDA will be added later if enabled.
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
# CUDA configuration — modern CMake approach
# ---------------------------------------------------------------------------
if(ENABLE_CUDA)
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)

    message("--- CUDA ENABLED (toolkit: ${CUDAToolkit_VERSION})")
    add_definitions(-DCUDA_ENABLED -DUSE_CUDA_NEIGHBORS)

    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)

    # Target GPU architecture — default is sm_75 (Tesla T4 / Colab)
    # Adjust for your hardware:
    #   A100  -> 80,  V100 -> 70,  RTX 30xx -> 86,  RTX 40xx -> 89
    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
        set(CMAKE_CUDA_ARCHITECTURES 75)
    endif()

    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 --use_fast_math")
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
    instance/Saving.hpp

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
    # movegen/MoveGenerators.cpp   <--- REMOVED because it does not exist (header-only)

    opt/routemin.hpp
    opt/RuinAndRecreate.hpp
    opt/SimulatedAnnealing.hpp
    opt/bpp.hpp

    solution/savings.hpp
    solution/Solution.hpp
    solution/Solution.cpp
)

# ---------------------------------------------------------------------------
# CUDA‑specific sources (appended only when CUDA is enabled)
# ---------------------------------------------------------------------------
if(ENABLE_CUDA)
    list(APPEND SOURCES
        cuda/CudaNeighborFinder.cu
        cuda/CudaNeighborFinder.hpp
        cuda/uniform_grid.cu       # uniform‑grid implementation
        cuda/uniform_grid.cuh      # header
        cuda/SavingsKernel.cu      # new GPU savings compute + sort
        cuda/SavingsKernel.cuh     # header
    )
endif()

# ---------------------------------------------------------------------------
# Executable target
# ---------------------------------------------------------------------------
add_executable(filo2 ${SOURCES})
set_target_properties(filo2 PROPERTIES OUTPUT_NAME "filo2-HYBRID-CPU-GPU")

# ---------------------------------------------------------------------------
# Include directories
# ---------------------------------------------------------------------------
# Add the project root so that includes like "cuda/uniform_grid.cuh" work.
target_include_directories(filo2 PRIVATE ${CMAKE_SOURCE_DIR})

# Also add CUDA toolkit include directories (if CUDA is enabled)
if(ENABLE_CUDA)
    target_include_directories(filo2 PRIVATE ${CUDAToolkit_INCLUDE_DIRS})
endif()

# ---------------------------------------------------------------------------
# Link libraries
# ---------------------------------------------------------------------------
target_link_libraries(filo2 PUBLIC ${LIBRARIES} OpenMP::OpenMP_CXX)

if(ENABLE_CUDA)
    target_link_libraries(filo2 PUBLIC CUDA::cudart)
    # Compile device code as relocatable for linking with host objects
    set_target_properties(filo2 PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
endif()

# OpenMP compile flag fallback for older CMake versions
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    target_compile_options(filo2 PRIVATE -fopenmp)
endif()
