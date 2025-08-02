
// Copyright 2025 Manoj Velmurugan
// SPDX-License-Identifier: MIT

#include <cuda_runtime.h>
#include "interp.hpp"

__global__ void bilinear_interp_kernel(const FLOAT *input, FLOAT *output, int in_h, int in_w, int out_h, int out_w, int channels)
{
    // marked constants to accidentally avoid modifying them
    const int idx_c = threadIdx.x + blockIdx.x * blockDim.x; // channel
    const int out_y = threadIdx.y + blockIdx.y * blockDim.y; // output row
    const int out_x = threadIdx.z + blockIdx.z * blockDim.z; // output col

    if (out_y >= out_h || out_x >= out_w || idx_c >= channels)
    {
        return;
    }

    // Calculate scaling factors
    const float scale_y = static_cast<float>(in_h) / static_cast<float>(out_h);
    const float scale_x = static_cast<float>(in_w) / static_cast<float>(out_w);

    // Map output coordinates to input coordinates using the "align centers" method
    const float src_y = (static_cast<float>(out_y) + 0.5f) * scale_y - 0.5f;
    const float src_x = (static_cast<float>(out_x) + 0.5f) * scale_x - 0.5f;

    // Get the top-left corner for interpolation
    const int y1 = floorf(src_y);
    const int x1 = floorf(src_x);

    // This dx and dy will be the fractional part, between 0 and 1
    const float dy = src_y - y1;
    const float dx = src_x - x1;

    // Get the four neighboring pixel coordinates, clamping to image boundaries
    const int y1_c = fmaxf(0, fminf(in_h - 1, y1));
    const int x1_c = fmaxf(0, fminf(in_w - 1, x1));
    const int y2_c = fmaxf(0, fminf(in_h - 1, y1 + 1));
    const int x2_c = fmaxf(0, fminf(in_w - 1, x1 + 1));
    
    // Get the channel offset for the input tensor
    const int poffset = idx_c * in_h * in_w;

    // Fetch the four corner pixel values
    const float tl = input[poffset + y1_c * in_w + x1_c];
    const float tr = input[poffset + y1_c * in_w + x2_c];
    const float bl = input[poffset + y2_c * in_w + x1_c];
    const float br = input[poffset + y2_c * in_w + x2_c];

    // Perform bilinear interpolation
    const float top = tl * (1.0f - dx) + tr * dx;
    const float bottom = bl * (1.0f - dx) + br * dx;
    const float result = top * (1.0f - dy) + bottom * dy;
 
    const int out_idx = idx_c * (out_h * out_w) + out_y * out_w + out_x;
    output[out_idx] = result;
}

DevicePointer<FLOAT> &BilinearInterp2D::forward(const DevicePointer<FLOAT> &input_device)
{
    std::vector<int> expected_shape = {input_prop.channels, input_prop.height, input_prop.width};
    auto actual_shape = input_device.get_shape();

    if (actual_shape != expected_shape) throw std::runtime_error("BilinearInterp2D: shape mismatch");

    const int TC = 8;
    dim3 threadcount(TC, TC, TC);
    dim3 blocks((output_prop.channels + TC - 1) / TC,
                (output_prop.height + TC - 1) / TC,
                (output_prop.width + TC - 1) / TC);

    std::cout << "starting bilinear interp kernel " << input_prop << " -> " << output_prop<< " blocks: " << blocks.x << " " << blocks.y << " " << blocks.z << std::endl;

    bilinear_interp_kernel<<<blocks, threadcount>>>(input_device.get(), output_device.get(), input_prop.height, input_prop.width, output_prop.height, output_prop.width, input_prop.channels);

    cudaDeviceSynchronize();

    return output_device;
}

DevicePointer<FLOAT> &BilinearInterp2D::get_output()
{
    return output_device;
}

BilinearInterp2D::BilinearInterp2D(ImgProperty input_prop_, int target_height_, int target_width_) : input_prop(input_prop_), target_height(target_height_), target_width(target_width_)
{
    if (target_height <= 0 || target_width <= 0)
        throw std::invalid_argument("target dimensions must be positive");

    output_prop.channels = input_prop.channels; // Same channels
    output_prop.height   = target_height;
    output_prop.width    = target_width;

    std::vector<int> output_shape = {output_prop.channels, output_prop.height, output_prop.width};
    output_device.alloc(output_shape);
}
