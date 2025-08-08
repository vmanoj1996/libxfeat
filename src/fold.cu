/*
Fold and unfold operations for xfeat network
input 1xHxW
output 64x(H/8)x(w/8)

and vice versa

nvcc -std=c++20 -arch=sm_89 fold.cu && ./a.out
*/

// Copyright 2025 Manoj Velmurugan
// SPDX-License-Identifier: MIT

#include <cuda_runtime.h>
#include <cassert>
#include <cstdio>
#include <iostream>

#include "fold.hpp"

__global__ void fold_kernel(const FLOAT *input_device, FLOAT *output_device, int height, int width, int ratio)
{
    // input idx
    int idx1 = threadIdx.x + blockIdx.x * blockDim.x;
    int idx2 = threadIdx.y + blockIdx.y * blockDim.y;

    if (idx1 < height && idx2 < width)
    {
        // compute the output dimension and index corresponding to idx1, idx2 inputs
        int idx0_out = (idx1 % ratio) * ratio + (idx2 % ratio);
        int idx1_out = idx1 / ratio;
        int idx2_out = idx2 / ratio;

        // int idx0_dim = ratio*ratio;
        int idx1_dim = height / ratio;
        int idx2_dim = width / ratio;

        output_device[idx0_out * (idx1_dim * idx2_dim) + idx1_out * idx2_dim + idx2_out] = input_device[idx1 * width + idx2];
    }
}


Fold2D::Fold2D(int height_, int width_, int ratio_, cudaStream_t stream_): Fold2D_common(height_, width_, ratio_)
{
    stream = stream_;
    if (height % reduction_ratio != 0 || width % reduction_ratio != 0)
    {
        std::cerr << "height and width should be multiple of " << reduction_ratio << std::endl;
        exit(1);
    }
    input_prop.channels = 1;
    input_prop.height = height;
    input_prop.width  = width;

    output_prop.channels = reduction_ratio*reduction_ratio;
    output_prop.height = height/reduction_ratio;
    output_prop.width  = width/reduction_ratio;

    output_device.alloc({output_prop.channels, output_prop.height, output_prop.width}); // alloc the storage. automagically checked and cleared by the destructor
}


DevicePointer<FLOAT>& Fold2D::forward(const DevicePointer<FLOAT>& input_device)
    {
        if (input_device.get() == output_device.get())
        {
            if(output_device.get() == nullptr) std::cout<<"Null ptr for some reason in fold forward\n";

            throw std::invalid_argument("Input and output buffers cannot be the same in fold");
        }

        const int TC = 16;
        dim3 threads(TC, TC);
        dim3 blocks((height + TC - 1) / TC, (width + TC - 1) / TC);

        fold_kernel<<<blocks, threads, 0, stream>>>(input_device.get(), output_device.get(), height, width, reduction_ratio);

        cudaDeviceSynchronize();

        return output_device;
    }

__global__ void unfold_kernel(const FLOAT *input_device, FLOAT *output_device, int height, int width, int ratio)
{
    // OUTPUT INDEX
    int idx1 = threadIdx.x + blockIdx.x * blockDim.x;
    int idx2 = threadIdx.y + blockIdx.y * blockDim.y;

    if (idx1 < height && idx2 < width)
    {
        // INPUT INDEX
        int subIndex1 = (idx1 % ratio);
        int subIndex2 = (idx2 % ratio);

        int idx0_in =  subIndex1 * ratio + subIndex2;
        int idx1_in = idx1 / ratio;
        int idx2_in = idx2 / ratio;

        // INPUT DIMENSIONS
        // int idx0_dim = ratio*ratio + 1;
        int idx1_dim = height / ratio;
        int idx2_dim = width / ratio;

        int input_idx = idx0_in * (idx1_dim * idx2_dim) + idx1_in * idx2_dim + idx2_in;
        output_device[idx1 * width + idx2] = input_device[input_idx];
    }
}


UnFold2D::UnFold2D(int height_, int width_, int ratio_, cudaStream_t stream_): Fold2D_common(height_, width_, ratio_)
{
    stream = stream_;
    
    if (height % reduction_ratio != 0 || width % reduction_ratio != 0)
    {
        std::cerr << "height and width should be multiple of " << reduction_ratio << std::endl;
        exit(1);
    }
    input_prop.channels = reduction_ratio*reduction_ratio + 1;
    input_prop.height = height/reduction_ratio;
    input_prop.width  = width/reduction_ratio;

    output_prop.channels = 1;
    output_prop.height = height;
    output_prop.width  = width;

    output_device.alloc({output_prop.channels, output_prop.height, output_prop.width}); // alloc the storage. automagically checked and cleared by the destructor
}

DevicePointer<FLOAT>& UnFold2D::forward(const DevicePointer<FLOAT>& input_device)
{
    if (input_device.get() == output_device.get())
    {
        if(output_device.get() == nullptr) std::cout<<"Null ptr for some reason\n";

        throw std::invalid_argument("Input and output buffers cannot be the same in unfold");
    }
    const int TC = 16;
    dim3 threads(TC, TC);
    dim3 blocks((height + TC - 1) / TC, (width + TC - 1) / TC);

    unfold_kernel<<<blocks, threads, 0, stream>>>(input_device.get(), output_device.get(), height, width, reduction_ratio);
    cudaDeviceSynchronize();

    return output_device;
}
