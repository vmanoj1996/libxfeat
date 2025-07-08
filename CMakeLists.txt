cmake_minimum_required(VERSION 3.18)
project(matx_test LANGUAGES CXX CUDA)

find_package(CUDAToolkit REQUIRED)
find_package(pybind11 REQUIRED)
find_package(matx REQUIRED)

set(CMAKE_CUDA_ARCHITECTURES 89)

add_executable(matx_test test/matx_test.cu)
target_link_libraries(matx_test matx::matx)

