
// Copyright 2025 Manoj Velmurugan
// SPDX-License-Identifier: MIT

#include <cuda_runtime.h>
#include "interp.hpp"

__global__ void bilinear_interp_kernel(const FLOAT *input, FLOAT *output, int in_h, int in_w, int out_h, int out_w, int channels)
{
    int idx_c = threadIdx.x + blockIdx.x * blockDim.x; // channel
    int out_y = threadIdx.y + blockIdx.y * blockDim.y; // output row
    int out_x = threadIdx.z + blockIdx.z * blockDim.z; // output col

    if (out_y >= out_h || out_x >= out_w || idx_c >= channels)
    {
        return;
    }

    // -0.5 to 0.5
    float out_y_norm = ((float)out_y / (float)out_h) - 0.5;
    float out_x_norm = ((float)out_x / (float)out_w) - 0.5;

    // printf("bilinear: %d %d %d\n", idx_c, out_y, out_x);

    // Map output coordinates to input coordinates. recheck this calculations
    float src_y = out_y_norm * in_h + in_h / 2;
    float src_x = out_x_norm * in_w + in_w / 2;

    src_y = (src_y >= 0) ? src_y : 1e-5;
    src_x = (src_x >= 0) ? src_x : 1e-5;

    // get the left top corner
    int y1 = floorf(src_y);
    int x1 = floorf(src_x);

    // limit within the valid image region
    y1 = (y1 >= 0) ? y1 : 0;
    x1 = (x1 >= 0) ? x1 : 0;

    y1 = (y1 < in_h) ? y1 : in_h - 1;
    x1 = (x1 < in_w) ? x1 : in_w - 1;

    int y2 = y1 + 1;
    int x2 = x1 + 1;

    // limit the second index
    y2 = (y2 < in_h) ? y2 : in_h - 1;
    x2 = (x2 < in_w) ? x2 : in_w - 1;

    // This dx and dy will be under 1
    float dy = src_y - y1;
    float dx = src_x - x1;

    int poffset = idx_c * in_h * in_w;

    // float tl = input[poffset + y1 * in_w + x1];
    // float tr = input[poffset + y1 * in_w + x2];
    // float bl = input[poffset + y2 * in_w + x1];
    // float br = input[poffset + y2 * in_w + x2];

    float tl = input[0];
    float tr = input[0];
    float bl = input[0];
    float br = input[0];

    float top = tl * (1.0f - dx) + tr * dx;
    float bottom = bl * (1.0f - dx) + br * dx;
    float result = top * (1.0f - dy) + bottom * dy;

    // printf("bilinear: %f %f %f\n", top, bottom, result);

    // Write to output
    int out_idx = idx_c * (out_h * out_w) + out_y * out_w + out_x;

    // printf("bilinear: %d \n", out_idx);

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
