# Copyright 2025 Manoj Velmurugan
# SPDX-License-Identifier: MIT

# Setup ------------------------------------------------------------------------------------------

cmake_minimum_required(VERSION 3.18)
project(cuda_conv2d LANGUAGES CXX CUDA)

find_package(CUDAToolkit REQUIRED)
find_package(HDF5 REQUIRED COMPONENTS C)
find_package(OpenCV REQUIRED)
find_package(Boost REQUIRED COMPONENTS system)
find_package(PkgConfig REQUIRED) 
pkg_check_modules(MATIO REQUIRED matio)

set(CMAKE_CUDA_ARCHITECTURES 89)
set(CMAKE_CXX_STANDARD 23)

# Set common flags that apply to all build types
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler -Wfatal-errors -Wreturn-local-addr")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wfatal-errors")

# --- Default Build Type and Optimization Flags ---

# Set Release as the default build type if the user doesn't specify one
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the build type" FORCE)
endif()

# For Release builds (e.g., cmake .. OR cmake -DCMAKE_BUILD_TYPE=Release ..)
# -O3 is for high optimization. -DNDEBUG disables asserts.
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG" CACHE STRING "Flags used for Release build")
set(CMAKE_CUDA_FLAGS_RELEASE "-O3 -DNDEBUG" CACHE STRING "Flags used for Release build")

# For Debug builds (e.g., cmake -DCMAKE_BUILD_TYPE=Debug ..)
# -O0 disables optimization. -g adds debug symbols for GDB.
set(CMAKE_CXX_FLAGS_DEBUG "-O0 -g" CACHE STRING "Flags used for Debug build")
set(CMAKE_CUDA_FLAGS_DEBUG "-O0 -g" CACHE STRING "Flags used for Debug build")


# Source --------------------------------------------------------------------------------------------

# Source files
file(GLOB_RECURSE SOURCE_FILES 
    ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cu
)

# Create source library
add_library(conv2d_lib ${SOURCE_FILES})

target_include_directories(conv2d_lib PUBLIC 
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${HDF5_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
)
target_link_libraries(conv2d_lib PUBLIC 
    CUDA::cudart
    ${HDF5_LIBRARIES}
    ${OpenCV_LIBS}
)

# This uses a "generator expression" to add the flag only for the Debug config.
target_compile_definitions(conv2d_lib PUBLIC 
    ${HDF5_DEFINITIONS}
    $<$<CONFIG:Debug>:ENABLE_XFEAT_DEBUG>
)

set_target_properties(conv2d_lib PROPERTIES POSITION_INDEPENDENT_CODE ON)

# Enable testing ------------------------------------------------------------------------------------
enable_testing()

# Create a common test library
add_library(test_lib INTERFACE)
target_include_directories(test_lib INTERFACE 
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${HDF5_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
    ${MATIO_INCLUDE_DIRS}
)
target_link_libraries(test_lib INTERFACE 
    conv2d_lib 
    ${MATIO_LIBRARIES}
)

# C++ tests
set(cpp_tests
    test_sigmoid
    test_xfeat_run
    test_interp
    test_pool
    test_xfeat_output
)

# CUDA tests (files ending in .cu)
set(cu_tests
    test_fold_simple
    test_conv2d
)

# Separate the python tests into "core" and "intermediate"
set(python_core_tests
    test_sigmoid.py
    test_interp.py
    test_pool.py
    test_conv2d.py
    test_xfeat_output.py
)
set(python_inter_test test_xfeat_inter.py)

# --------------------------------------------------------------------------------------------------------------------------
# Build and add C++ tests
foreach(test ${cpp_tests})
    add_executable(${test} test/${test}.cpp)
    target_link_libraries(${test} test_lib)
    add_test(NAME ${test} COMMAND ${test})
    set_tests_properties(${test} PROPERTIES LABELS "cpp_tests")
endforeach()

# Build and add CUDA tests
foreach(test ${cu_tests})
    add_executable(${test} test/${test}.cu)
    target_link_libraries(${test} test_lib)
    add_test(NAME ${test} COMMAND ${test})
    set_tests_properties(${test} PROPERTIES LABELS "cpp_tests") # Label as a C++ test for grouping
endforeach()

# Add core Python tests (unconditionally)
foreach(test ${python_core_tests})
    add_test(NAME ${test} 
             COMMAND bash -c "source ~/miniconda3/etc/profile.d/conda.sh && conda activate cusfm && python3 ${CMAKE_CURRENT_SOURCE_DIR}/test/${test}"
    )
    set_tests_properties(${test} PROPERTIES LABELS "python_tests")
endforeach()

# CMAKE_BUILD_TYPE is case-insensitive, so we check for "Debug".
if(CMAKE_BUILD_TYPE MATCHES "^([Dd][Ee][Bb][Uu][Gg])$")
    message(STATUS "Debug build: 'test_xfeat_inter.py' will be included in the test suite.")
    add_test(NAME ${python_inter_test} 
             COMMAND bash -c "source ~/miniconda3/etc/profile.d/conda.sh && conda activate cusfm && python3 ${CMAKE_CURRENT_SOURCE_DIR}/test/${python_inter_test}"
    )
    set_tests_properties(${python_inter_test} PROPERTIES LABELS "python_tests")
else()
    message(STATUS "Release build: 'test_xfeat_inter.py' will be skipped.")
    # Add a dummy test that just prints a message and passes.
    add_test(NAME skip_${python_inter_test}
             COMMAND ${CMAKE_COMMAND} -E echo "INFO: Skipping test_xfeat_inter.py because this is a Release build."
    )
    set_tests_properties(skip_${python_inter_test} PROPERTIES LABELS "python_tests")
endif()

# --------------------------------------------------------------------------------------------------------------------------

# Custom target to run tests in order
add_custom_target(unit_tests
    COMMAND ${CMAKE_CTEST_COMMAND} -L cpp_tests --output-on-failure
    COMMAND ${CMAKE_CTEST_COMMAND} -L python_tests --output-on-failure
    COMMENT "Running all tests (C++ first, then Python)"
    VERBATIM
)