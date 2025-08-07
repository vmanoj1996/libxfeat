// Copyright 2025 Manoj Velmurugan
// SPDX-License-Identifier: MIT

// how to increase the stack size if needed
// g++ -Wl,--stack,16777216 program.cpp -o program
// g++ -fsanitize=address -g program.cpp -o program
// g++ -fsanitize=address,undefined,leak -g program.cpp

// nvcc -std=c++20 -arch=sm_89 conv2d.cu && ./a.out

#include <cuda_runtime.h>
#include <cassert>
#include <cstdio>
#include <stdexcept>
#include <vector>
#include <string>
#include "conv2d.hpp"
#include "device_ops.hpp"
#include <iostream>

template<typename Operation>
__global__ void convolve2d_kernel(const FLOAT * __restrict__ input_device, const FLOAT * __restrict__ kernel_device, FLOAT * __restrict__ output_device, Conv2DParams p, ImgProperty input_prop, ImgProperty output_prop, Operation op)
{
    extern __shared__ FLOAT kernel_per_ch[];

    int idx_co  = threadIdx.x + blockIdx.x * blockDim.x; // channel output
    int out_row = threadIdx.y + blockIdx.y * blockDim.y;
    int out_col = threadIdx.z + blockIdx.z * blockDim.z;
    
    // cooperative copy to copy the kernel to shared memory
    int tid = threadIdx.y * blockDim.z + threadIdx.z; // unique thread id (first channel id is 0 always)
    int total_threads = blockDim.y * blockDim.z; 
    int kernel_net_size = p.ci * p.k1 * p.k2;

    if (out_row < output_prop.height && out_col < output_prop.width && idx_co < p.co)
    {
        // copy from idx_co * kernel_net_size to idx_co * kernel_net_size + kernel_net_size (last excluded)
        // first thread works on 0, 256, 512 ... till the end of kernel_net_size
        // second thread works on 1, 257, 513, ... till the end
        // last thread works on total_thread-1, total_thread-1-256 and so on. so no elements are skipped in this copy pattern.
        // plus all threads in the block get to do some work to copy our giant kernel.
        // plus all threads would work on closeby memory regions (which is cache friendly)
        for(int i = tid; i < kernel_net_size; i += total_threads) 
        {
            kernel_per_ch[i] = kernel_device[idx_co * kernel_net_size + i];
        }
        __syncthreads();

        FLOAT sum = 0.0f;
        // once padded, the first operation that will happen is on this particular index in the imaginary padded input (implicit)
        int in_row_start = out_row * p.s1 - p.p1;
        int in_col_start = out_col * p.s2 - p.p2;

        for (int idx_ci = 0; idx_ci < p.ci; idx_ci++)
        {
            for (int kernel_row = 0; kernel_row < p.k1; kernel_row++)
            {
                int input_row_index = (in_row_start + kernel_row);
                bool row_valid = (input_row_index >= 0 && input_row_index < input_prop.height);

                for (int kernel_col = 0; kernel_col < p.k2; kernel_col++)
                {                    
                    // load kernel from shared memory for faster access
                    FLOAT kernel_value = kernel_per_ch[idx_ci * (p.k1 * p.k2) + kernel_row * (p.k2) + kernel_col];

                    int input_col_index = (in_col_start + kernel_col);
                    bool col_valid = (input_col_index >= 0 && input_col_index < input_prop.width);

                    FLOAT input_value = (row_valid && col_valid)? input_device[idx_ci * input_prop.height * input_prop.width + input_row_index * input_prop.width + input_col_index]: 0.0f;

                    sum += input_value * kernel_value;
                }
            }
        }

        int o_index = idx_co * output_prop.height * output_prop.width + out_row * output_prop.width + out_col;

        output_device[o_index] = op.forward(sum, idx_co);
    }
}

template<typename Operation>
Conv2D<Operation>::Conv2D(ImgProperty input_prop_, Conv2DParams params_, const std::vector<FLOAT> &kernel_data, Operation post_op_): 
    params(params_), input_prop(input_prop_), post_op(post_op_)
{
    validate_params();
    output_prop.channels = params.co;
    output_prop.height = (input_prop.height + 2 * params.p1 - params.k1) / params.s1 + 1;
    output_prop.width =  (input_prop.width + 2 * params.p2 - params.k2) / params.s2 + 1;

    std::vector<int> output_Shape = {params.co, output_prop.height, output_prop.width};
    output_device.alloc(output_Shape);

    std::vector<int> kernel_Shape = {params.co, params.ci, params.k1, params.k2};
    kernel_device.alloc(kernel_Shape);

    // set the kernel up
    set_kernel(kernel_data);
}

template<typename Operation>
Conv2D<Operation>::~Conv2D()
{
    post_op.destroy();
}

template<typename Operation>
DevicePointer<FLOAT> &Conv2D<Operation>::forward(const DevicePointer<FLOAT> &input_device)
{
    std::vector<int> expected_shape = {input_prop.channels, input_prop.height, input_prop.width};
    auto actual_shape = input_device.get_shape();

    if (actual_shape != expected_shape) throw std::runtime_error("conv2d: shape mismatch");

    // spawn just 1 thread per output channel. but it is still effective. the 3d indexing is just for convenience. 
    // each exec block would correspond to 16x16 output tile for a single output channel
    const int TC = 16;
    dim3 threadcount(1, TC, TC);
    dim3 blocks(params.co, (output_prop.height + TC - 1) / TC, (output_prop.width + TC - 1) / TC);

    const int kernel_shared_per_block = params.k1 * params.k2 * params.ci * sizeof(FLOAT);

#ifdef ENABLE_XFEAT_DEBUG
    std::cout<<"starting conv kernel "<<input_prop<<" "<<output_prop<<" "<<params<<blocks.x<<" "<<blocks.y<<" "<<blocks.z<<" "<<std::endl;
#endif
    convolve2d_kernel<<<blocks, threadcount, kernel_shared_per_block>>>(input_device.get(), kernel_device.get(), output_device.get(), params, input_prop, output_prop, post_op);
    cudaDeviceSynchronize();

    return output_device;
}

template<typename Operation>
void Conv2D<Operation>::set_kernel(const std::vector<FLOAT> &kernel_data)
{
    size_t expected_size = params.co * params.ci * params.k1 * params.k2;
    if (kernel_data.size() != expected_size)
    {
        throw std::invalid_argument("Kernel size mismatch: expected " + std::to_string(expected_size) + " weights, got " + std::to_string(kernel_data.size()));
    }
    cudaMemcpy(kernel_device.get(), kernel_data.data(), expected_size * sizeof(FLOAT), cudaMemcpyHostToDevice);
}

template<typename Operation>
DevicePointer<FLOAT> &Conv2D<Operation>::get_output()
{
    return output_device;
}

template<typename Operation>
Conv2DParams Conv2D<Operation>::get_param() const
{
    return params;
}

template<typename Operation>
void Conv2D<Operation>::validate_params()
{
    {
        if (params.k1 % 2 == 0)
        throw std::invalid_argument("k1 must be odd");
    }
    if (params.k2 % 2 == 0)
    {
        throw std::invalid_argument("k2 must be odd");
    }
    if (params.s1 <= 0)
    {
        throw std::invalid_argument("s1 (stride height) must be positive");
    }
    if (params.s2 <= 0)
    {
        throw std::invalid_argument("s2 (stride width) must be positive");
    }
    if (params.p1 < 0)
    {
        throw std::invalid_argument("p1 (padding height) must be non-negative");
    }
    if (params.p2 < 0)
    {
        throw std::invalid_argument("p2 (padding width) must be non-negative");
    }
}

template class Conv2D<BatchNormRelu>;
template class Conv2D<Identity>;
template class Conv2D<BiasOp>;

#ifdef ACTIVATE_CONV_MAIN
int main()
{
    ImgProperty input_prop = {40, 60};                    // height, width
    Conv2DParams conv_params = {1, 1, 3, 16, 2, 2, 1, 1}; // k1,k2,ci,co,s1,s2,p1,p2

    Conv2D convlayer(input_prop, conv_params);

    float *input_device;
    // Fix: use conv_params values instead of undefined variables
    cudaMalloc(&input_device, conv_params.ci * input_prop.height * input_prop.width * sizeof(float));
    cudaMemset(input_device, 0, conv_params.ci * input_prop.height * input_prop.width * sizeof(float));

    convlayer.forward(input_device);

    cudaFree(input_device);
    return 0;
}
#endif