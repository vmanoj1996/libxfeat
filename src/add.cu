// Copyright 2025 Manoj Velmurugan
// SPDX-License-Identifier: MIT

#include "add.hpp"
#include <stdexcept>
#include <cuda_runtime.h>

// CUDA kernel for element-wise addition (2 or 3 inputs)
template<bool HAS_THREE_INPUTS>
__global__ void elementwise_add_kernel(const float* __restrict__ input1, const float* __restrict__ input2, const float* __restrict__ input3, float* __restrict__ output, int total_size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_size) return;

    if constexpr (HAS_THREE_INPUTS) 
    {
        output[idx] = input1[idx] + input2[idx] + input3[idx];
    } else 
    {
        output[idx] = input1[idx] + input2[idx];
    }

}

Add::Add(ImgProperty output_prop_, cudaStream_t stream_) : output_prop(output_prop_), input_prop(output_prop_)
{
    stream = stream_;
    std::vector<int> shape = {output_prop.channels, output_prop.height, output_prop.width};
    output_device.alloc(shape);
}

DevicePointer<FLOAT>& Add::forward(const std::vector<const DevicePointer<FLOAT>*>& inputs)
{
    if (inputs.size() < 2 || inputs.size() > 3) {
        throw std::invalid_argument("Add layer expects 2 or 3 inputs");
    }
    
    int total_size = output_prop.channels * output_prop.height * output_prop.width;
    dim3 block(256);
    dim3 grid((total_size + block.x - 1) / block.x);
    
    if (inputs.size() == 3) 
    {
        elementwise_add_kernel<true><<<grid, block, 0, stream>>>(inputs[0]->get(), inputs[1]->get(), inputs[2]->get(), output_device.get(), total_size);
    } else 
    {
        elementwise_add_kernel<false><<<grid, block, 0, stream>>>(inputs[0]->get(), inputs[1]->get(), nullptr, output_device.get(), total_size);
    }
    
    CUDA_SYNC_IF_NEEDED();
    return output_device;
}

DevicePointer<FLOAT>& Add::forward(const DevicePointer<FLOAT>& input)
{
    throw std::runtime_error("Add layer requires multiple inputs. Use forward(vector<inputs>) instead.");
}
