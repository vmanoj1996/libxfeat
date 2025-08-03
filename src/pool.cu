
// Copyright 2025 Manoj Velmurugan
// SPDX-License-Identifier: MIT

#include <cuda_runtime.h>
#include <cassert>
#include <cstdio>
#include <stdexcept>
#include <vector>
#include <string>
#include "pool.hpp"
#include "device_ops.hpp"
#include <iostream>

__global__ void avgpool2d_kernel(const FLOAT *input_device, FLOAT *output_device, PoolParams p, ImgProperty input_prop, ImgProperty output_prop)
{
    /* Parameter documentation:


    */
    int idx_co = threadIdx.x + blockIdx.x * blockDim.x; // channel output
    int out_row = threadIdx.y + blockIdx.y * blockDim.y;
    int out_col = threadIdx.z + blockIdx.z * blockDim.z;

    float inv_pool_count = 1.0f/(p.k1*p.k2);

    if (out_row < output_prop.height && out_col < output_prop.width && idx_co < output_prop.channels)
    {
        FLOAT sum = 0.0f;

        // once padded, the first operation that will happen is on this particular index in the imaginary padded input (implicit)
        int in_row_start = out_row * p.s1 - p.p1;
        int in_col_start = out_col * p.s2 - p.p2;

        int idx_ci = idx_co;
        for (int kernel_row = 0; kernel_row < p.k1; kernel_row++)
        {
            for (int kernel_col = 0; kernel_col < p.k2; kernel_col++)
            {
                FLOAT input_value = ((in_row_start + kernel_row) >= 0 && (in_row_start + kernel_row) < input_prop.height &&
                                        (in_col_start + kernel_col) >= 0 && (in_col_start + kernel_col) < input_prop.width)
                                        ? input_device[idx_ci * input_prop.height * input_prop.width + (in_row_start + kernel_row) * input_prop.width + (in_col_start + kernel_col)]
                                        : 0.0f;

                sum += input_value;
            }
        }


        int o_index = idx_co * output_prop.height * output_prop.width + out_row * output_prop.width + out_col;

        output_device[o_index] = sum*inv_pool_count;
    }
}

AvgPool2D::AvgPool2D(ImgProperty input_prop_, PoolParams params_) : params(params_), input_prop(input_prop_)
{
    if (params.k1 <= 0 || params.k2 <= 0) throw std::invalid_argument("kernel size must be positive");
    if (params.s1 <= 0 || params.s2 <= 0) throw std::invalid_argument("stride must be positive");
    if (params.p1 < 0 || params.p2 < 0)   throw std::invalid_argument("padding must be non-negative");

    output_prop.channels = input_prop.channels;
    output_prop.height = (input_prop.height + 2 * params.p1 - params.k1) / params.s1 + 1;
    output_prop.width = (input_prop.width + 2 * params.p2 - params.k2) / params.s2 + 1;

    std::vector<int> output_Shape = {output_prop.channels, output_prop.height, output_prop.width};
    output_device.alloc(output_Shape);
}

DevicePointer<FLOAT> &AvgPool2D::forward(const DevicePointer<FLOAT> &input_device)
{
    const int TC = 8;
    dim3 threadcount(TC, TC, TC);

    dim3 blocks((output_prop.channels + TC - 1) / TC,
                (output_prop.height   + TC - 1) / TC,
                (output_prop.width    + TC - 1) / TC);

#ifdef ENABLE_XFEAT_DEBUG
    std::cout << "starting pool kernel " << input_prop << " " << output_prop << " "<< blocks.x << " " << blocks.y << " " << blocks.z << " " << std::endl;
#endif
    avgpool2d_kernel<<<blocks, threadcount>>>(input_device.get(), output_device.get(), params, input_prop, output_prop);
    cudaDeviceSynchronize();

    return output_device;
}
