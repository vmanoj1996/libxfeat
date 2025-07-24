# Setup ------------------------------------------------------------------------------------------
cmake_minimum_required(VERSION 3.18)
project(cuda_conv2d LANGUAGES CXX CUDA)

find_package(CUDAToolkit REQUIRED)
find_package(HDF5 REQUIRED COMPONENTS C)
find_package(OpenCV REQUIRED)


set(CMAKE_CUDA_ARCHITECTURES 89)
set(CMAKE_CXX_STANDARD 20)

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

target_compile_definitions(conv2d_lib PUBLIC ${HDF5_DEFINITIONS})
set_target_properties(conv2d_lib PROPERTIES POSITION_INDEPENDENT_CODE ON)

# Add executable for XFeat main
add_executable(xfeat_main src/xfeat.cpp)  # or wherever your xfeat file is
target_compile_definitions(xfeat_main PRIVATE ACTIVATE_XFEAT_MAIN)
target_link_libraries(xfeat_main conv2d_lib)
