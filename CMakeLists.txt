cmake_minimum_required(VERSION 3.18)
project(test_matx LANGUAGES CXX CUDA)

find_package(CUDAToolkit REQUIRED)
find_package(pybind11 REQUIRED)
find_package(matx REQUIRED)

set(CMAKE_CUDA_ARCHITECTURES 89)

# Enable testing
enable_testing()

# C++ tests
set(cpp_tests
   test_matx
   test_matx_blur
   benchmark_matx
   # Add more C++ tests here
)

# Python tests  
set(python_tests
   test_xfeat.py
   benchmark_torch.py
   # Add more Python tests here
)

# Create a common test library
add_library(test_lib INTERFACE)
target_link_libraries(test_lib INTERFACE matx::matx)

# Build C++ test executables
foreach(test ${cpp_tests})
   add_executable(${test} test/${test}.cu)
   target_link_libraries(${test} test_lib)
   add_test(NAME ${test} COMMAND ${test})
endforeach()

# Add Python tests with conda environment
foreach(test ${python_tests})
   add_test(NAME ${test} 
            COMMAND bash -c "source ~/miniconda3/etc/profile.d/conda.sh && conda activate cusfm && python3 ${CMAKE_CURRENT_SOURCE_DIR}/test/${test}"
            )
endforeach()
