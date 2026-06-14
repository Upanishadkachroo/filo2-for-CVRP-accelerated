cmake_minimum_required(VERSION 3.22)
project(filo2)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Compiler options
set(WARNING_FLAGS "-Wall -Wextra -Wpedantic -Wuninitialized")
set(OPT_FLAGS "-O3 -march=native -flto=auto")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${WARNING_FLAGS}")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -fsanitize=address -fsanitize=undefined -fsanitize-address-use-after-scope -fno-omit-frame-pointer -g")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} ${OPT_FLAGS}")

set(CMAKE_VERBOSE_MAKEFILE ON)

add_definitions(-DUSE_CUDA_NEIGHBORS)
add_definitions(-DUSE_GRID_NEIGHBOR)   # enable the grid-based neighbor finder

# Options
option(ENABLE_VERBOSE "Enable verbose output" ON)
option(ENABLE_GUI "Enable graphical interface" OFF)
option(ENABLE_TIMELIMIT "Enable time limit termination" OFF)
option(ENABLE_CUDA "Enable CUDA support" OFF)

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

# OpenMP (mandatory for parallel preprocessing)
find_package(OpenMP REQUIRED)

# CUDA configuration (modern)
if(ENABLE_CUDA)
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)
    message("--- CUDA ENABLED")
    add_definitions(-DCUDA_ENABLED -DUSE_CUDA_NEIGHBORS)

    # Set CUDA architecture (adjust to your GPU, e.g., sm_61 for GTX 1080, sm_75 for RTX 2080, sm_86 for RTX 3080)
    set(CUDA_ARCH "sm_75")   # for Tesla T4 (also works for V100, RTX 20xx)
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 -arch=${CUDA_ARCH} --use_fast_math")
    set(CMAKE_CUDA_HOST_COMPILER ${CMAKE_CXX_COMPILER})
else()
    message("--- CUDA DISABLED")
endif()

# Collect source files (explicit list)
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

# Add CUDA sources if enabled
if(ENABLE_CUDA)
    list(APPEND SOURCES
        cuda/CudaNeighborFinder.cu
        cuda/CudaNeighborFinder.hpp
    )
endif()

# Add executable
add_executable(filo2 ${SOURCES})
set_target_properties(filo2 PROPERTIES OUTPUT_NAME "filo2-HYBRID-CPU-GPU")

# Link libraries
target_link_libraries(filo2 PUBLIC ${LIBRARIES} OpenMP::OpenMP_CXX)

if(ENABLE_CUDA)
    # Link CUDA runtime library (modern target)
    target_link_libraries(filo2 PUBLIC CUDA::cudart)
endif()

# Ensure OpenMP flags are passed even if find_package is not perfect (fallback)
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    target_compile_options(filo2 PRIVATE -fopenmp)
endif()

if(ENABLE_CUDA)
    list(APPEND SOURCES
        cuda/CudaNeighborFinder.cu
        cuda/CudaNeighborFinder.hpp
        cuda/GridNeighborFinder.cu
        cuda/GridNeighborFinder.hpp
    )
    add_definitions(-DUSE_CUDA_NEIGHBORS)
    # Optionally enable grid method
    add_definitions(-DUSE_GRID_NEIGHBOR)
endif()