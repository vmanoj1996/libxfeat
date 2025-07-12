# Setup ------------------------------------------------------------------------------------------
cmake_minimum_required(VERSION 3.18)
project(test_matx LANGUAGES CXX CUDA)

find_package(CUDAToolkit REQUIRED)
find_package(pybind11 REQUIRED)
find_package(matx REQUIRED)
find_package(HDF5 REQUIRED COMPONENTS C)

set(CMAKE_CUDA_ARCHITECTURES 89)

# Source --------------------------------------------------------------------------------------------

# Source files
file(GLOB_RECURSE SOURCE_FILES 
    ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cu
)

# Create source library
add_library(src_lib ${SOURCE_FILES})
target_include_directories(src_lib PUBLIC 
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${HDF5_INCLUDE_DIRS}
)
target_link_libraries(src_lib PUBLIC 
   matx::matx
   ${HDF5_LIBRARIES}
)

target_compile_definitions(src_lib PUBLIC ${HDF5_DEFINITIONS})

# Enable testing ------------------------------------------------------------------------------------
# Create a common test library
add_library(test_lib INTERFACE)
target_include_directories(test_lib INTERFACE 
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${HDF5_INCLUDE_DIRS}
)
target_link_libraries(test_lib INTERFACE src_lib)

enable_testing()

# C++ tests (regular unit tests)
set(cpp_tests
   test_matx
   test_matx_blur
   test_libxfeat
   test_xfeat_param
   # Add more C++ tests here
)

# Python tests  
set(python_tests
   test_xfeat.py
   # Add more Python tests here
)

# C++ benchmarks
set(cpp_benchmarks
   benchmark_matx
   # Add more C++ benchmarks here
)

# Python benchmarks
set(python_benchmarks
   benchmark_torch.py
   # Add more Python benchmarks here
)

# Build C++ test executables
foreach(test ${cpp_tests})
   add_executable(${test} test/${test}.cu)
   target_link_libraries(${test} test_lib)
   add_test(NAME ${test} COMMAND ${test})
   set_tests_properties(${test} PROPERTIES LABELS "unit_tests")
endforeach()

# Add Python tests with conda environment
foreach(test ${python_tests})
   add_test(NAME ${test} 
            COMMAND bash -c "source ~/miniconda3/etc/profile.d/conda.sh && conda activate cusfm && python3 ${CMAKE_CURRENT_SOURCE_DIR}/test/${test}"
            )
   set_tests_properties(${test} PROPERTIES LABELS "unit_tests")
endforeach()

# Build C++ benchmark executables
foreach(benchmark ${cpp_benchmarks})
   add_executable(${benchmark} test/${benchmark}.cu)
   target_link_libraries(${benchmark} test_lib)
   # Add as test but with benchmark label and longer timeout
   add_test(NAME ${benchmark} COMMAND ${benchmark})
   set_tests_properties(${benchmark} PROPERTIES 
       LABELS "benchmarks"
       TIMEOUT 300  # 5 minutes
   )
endforeach()

# Add Python benchmarks with conda environment
foreach(benchmark ${python_benchmarks})
   add_test(NAME ${benchmark} 
            COMMAND bash -c "source ~/miniconda3/etc/profile.d/conda.sh && conda activate cusfm && python3 ${CMAKE_CURRENT_SOURCE_DIR}/test/${benchmark}"
            )
   set_tests_properties(${benchmark} PROPERTIES 
       LABELS "benchmarks"
       TIMEOUT 600  # 10 minutes for Python benchmarks
   )
endforeach()

# Custom target to run only benchmarks
add_custom_target(benchmarks
    COMMAND ${CMAKE_CTEST_COMMAND} -L benchmarks --verbose
    COMMENT "Running benchmark tests"
    VERBATIM
)

# Custom target to run only unit tests (excluding benchmarks)
add_custom_target(unit_tests
    COMMAND ${CMAKE_CTEST_COMMAND} -L unit_tests --output-on-failure
    COMMENT "Running unit tests"
    VERBATIM
)