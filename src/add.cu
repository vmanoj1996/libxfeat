// Copyright 2025 Manoj Velmurugan
// SPDX-License-Identifier: MIT

#include "add.hpp"
#include <stdexcept>
#include <cuda_runtime.h>

// CUDA kernel for element-wise addition
__global__ void elementwise_add_kernel(const float* input1, const float* input2, const float* input3, 
                                      float* output, int total_size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total_size) {
        output[idx] = input1[idx] + input2[idx] + input3[idx];
    }
}

Add::Add(ImgProperty output_prop_) : output_prop(output_prop_), input_prop(output_prop_)
{
    std::vector<int> shape = {output_prop.channels, output_prop.height, output_prop.width};
    output_device.alloc(shape);
}

DevicePointer<FLOAT>& Add::forward(const std::vector<const DevicePointer<FLOAT>*>& inputs)
{
    if (inputs.size() != 3) {
        throw std::invalid_argument("Add layer expects exactly 3 inputs");
    }
    
    int total_size = output_prop.channels * output_prop.height * output_prop.width;
    
    // Launch kernel
    dim3 block(256);
    dim3 grid((total_size + block.x - 1) / block.x);
    
    elementwise_add_kernel<<<grid, block>>>(inputs[0]->get(), inputs[1]->get(), inputs[2]->get(), output_device.get(), total_size);
    cudaDeviceSynchronize();
    
    return output_device;
}

DevicePointer<FLOAT>& Add::forward(const DevicePointer<FLOAT>& input)
{
    throw std::runtime_error("Add layer requires multiple inputs. Use forward(vector<inputs>) instead.");
}
