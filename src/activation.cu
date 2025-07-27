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
__global__ void apply_activation_kernel(const float* input, float* output, int total_size, Operation op)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total_size) {
        output[idx] = op.forward(input[idx], idx);
    }
}

template<typename Operation>
const DevicePointer<FLOAT>& ActivationLayer<Operation>::forward(const DevicePointer<FLOAT>& input)
{
    int total_size = input_prop.channels * input_prop.height * input_prop.width;
    
    // Define grid and block dimensions
    dim3 block(256);
    dim3 grid((total_size + block.x - 1) / block.x);
    
    // Apply activation element-wise
    apply_activation_kernel<<<grid, block>>>(input.get(), output_device.get(), total_size, op);
    
    cudaDeviceSynchronize();
    return output_device;
}