cmake_minimum_required(VERSION 3.18)
project(test_matx LANGUAGES CXX CUDA)

find_package(CUDAToolkit REQUIRED)
find_package(pybind11 REQUIRED)
find_package(matx REQUIRED)

set(CMAKE_CUDA_ARCHITECTURES 89)

# Enable testing
enable_testing()

# List all test files explicitly
set(tests
   test_matx
   # Add more tests here as you create them
   # test_fft
   # test_tensor
   # test_operators
)

# Create a common test library with shared settings
add_library(test_lib INTERFACE)
target_link_libraries(test_lib INTERFACE matx::matx)

# Build each test executable and register with CTest
foreach(test ${tests})
   add_executable(${test} test/${test}.cu)
   target_link_libraries(${test} test_lib)
   add_test(NAME ${test} COMMAND ${test})
endforeach()