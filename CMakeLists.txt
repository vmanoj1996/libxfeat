# Setup ------------------------------------------------------------------------------------------
cmake_minimum_required(VERSION 3.18)
project(cuda_conv2d LANGUAGES CXX CUDA)

find_package(CUDAToolkit REQUIRED)
find_package(Matlab REQUIRED COMPONENTS MEX_COMPILER)

set(CMAKE_CUDA_ARCHITECTURES 89)
set(CMAKE_CXX_STANDARD 17)

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
    ${Matlab_INCLUDE_DIRS}  # Add MATLAB headers
)
target_link_libraries(conv2d_lib PUBLIC 
    CUDA::cudart
)

set_target_properties(conv2d_lib PROPERTIES POSITION_INDEPENDENT_CODE ON)

# MEX target
if(Matlab_FOUND)
    matlab_add_mex(NAME conv2d_mex 
                   SRC src/conv2d_mex.cu 
                   LINK_TO conv2d_lib)
    
    # Add static linking flags for libstdc++ and libgcc
    target_link_options(conv2d_mex PRIVATE 
        -static-libgcc 
        -static-libstdc++
    )
endif()