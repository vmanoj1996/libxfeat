// Copyright 2025 Manoj Velmurugan
// SPDX-License-Identifier: MIT

#include <cuda_runtime.h>
#include <cassert>
#include <cstdio>
#include <stdexcept>
#include <vector>
#include <string>
#include "activation.hpp"
#include "device_ops.hpp"
#include <iostream>
#include <primitives.hpp>
#include <memory>

// CUDA kernel for applying activation functions
template<typename Operation>
__global__ void apply_activation_kernel(const float* __restrict__ input, float* __restrict__ output, int total_size, const Operation op)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total_size) {
        output[idx] = op.forward(input[idx], idx);
    }
}

template<typename Operation>
DevicePointer<FLOAT>& ActivationLayer<Operation>::forward(const DevicePointer<FLOAT>& input)
{
    int total_size = input_prop.channels * input_prop.height * input_prop.width;
    
    // Define grid and block dimensions
    dim3 block(256);
    dim3 grid((total_size + block.x - 1) / block.x);
    
    // Apply activation element-wise
    apply_activation_kernel<<<grid, block, 0, stream>>>(input.get(), output_device.get(), total_size, op);
    
    CUDA_SYNC_IF_NEEDED();
    return output_device;
}