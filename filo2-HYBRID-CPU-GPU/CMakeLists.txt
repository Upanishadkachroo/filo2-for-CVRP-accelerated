cmake_minimum_required(VERSION 3.18)
project(filo2 CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(WARNING_FLAGS "-Wall -Wextra -Wpedantic -Wuninitialized")
set(OPT_FLAGS     "-O3 -march=native -flto=auto")
set(CMAKE_CXX_FLAGS         "${CMAKE_CXX_FLAGS} ${WARNING_FLAGS}")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} ${OPT_FLAGS}")
set(CMAKE_CXX_FLAGS_DEBUG
    "${CMAKE_CXX_FLAGS_DEBUG} -fsanitize=address -fsanitize=undefined \
     -fsanitize-address-use-after-scope -fno-omit-frame-pointer -g")
set(CMAKE_VERBOSE_MAKEFILE ON)

option(ENABLE_VERBOSE   "Enable verbose output"         ON)
option(ENABLE_GUI       "Enable graphical interface"    OFF)
option(ENABLE_TIMELIMIT "Enable time limit termination" OFF)

if(ENABLE_VERBOSE)
    message("--- Verbose output ENABLED")
    add_definitions(-DVERBOSE)
endif()
if(ENABLE_GUI)
    add_definitions(-DGUI)
    set(LIBRARIES ${LIBRARIES} glfw GL)
endif()
if(ENABLE_TIMELIMIT)
    add_definitions(-DTIMELIMIT)
endif()

find_package(OpenMP REQUIRED)
find_package(MPI    REQUIRED)

add_executable(filo2
    base/BinaryHeap.hpp
    base/BitMatrix.hpp
    base/FixedSizeValueStack.hpp
    base/Flat2DVector.hpp
    base/functor.hpp
    base/KDTree.cpp
    base/KDTree.hpp
    base/LRUCache.hpp
    base/macro.hpp
    base/NonCopyable.hpp
    base/PrettyPrinter.hpp
    base/SmallFlatMap.hpp
    base/SmallFlatSet.hpp
    base/SparseIntSet.hpp
    base/Timer.hpp
    base/VectorView.hpp
    base/Welford.hpp

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

    opt/bpp.hpp
    opt/routemin.hpp
    opt/RuinAndRecreate.hpp
    opt/SimulatedAnnealing.hpp

    solution/savings.hpp
    solution/SavingsImpl.hpp
    solution/Solution.cpp
    solution/Solution.hpp

    main.cpp
    Parameters.hpp
    Renderer.hpp
)

target_include_directories(filo2 PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${MPI_CXX_INCLUDE_DIRS}
)

target_link_libraries(filo2 PUBLIC
    ${LIBRARIES}
    OpenMP::OpenMP_CXX
    MPI::MPI_CXX
)

option(FILO2_BENCH "Enable per-step timing for benchmarking" OFF)
if(FILO2_BENCH)
    target_compile_definitions(filo2 PRIVATE BENCH)
    message(STATUS "BENCH timing enabled")
endif()
