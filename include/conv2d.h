// Simple convolve 2d (non batched)

/*
Requirements:
1. No batch needed
2. preserve the size of the input image
3. input is  ci x H1 x W1
4. output is co x H2 x W2
5. Parameter count is co x ci x k1 x k2

Pytorch saves the parameters for conv layers in this format according to claude. Verify this
[out_channels, in_channels, kernel_height, kernel_width]

k1 for row and k2 for column
*/

#pragma once

#include <vector>

using FLOAT = float;

struct Conv2DParams
{
    int k1, k2, ci, co;
    int s1, s2, p1, p2;
};

struct ImgProperty
{
    int height;
    int width;
};

class Convolve2D
{

private:
    FLOAT *kernel_device; // co, ci, k1, k2 order for cache optimality. Thats how pytorch is built too. optimized for row major operations
    FLOAT *output_device; // co x output_height x output_width

    Conv2DParams params;
    ImgProperty input_prop, output_prop;
    dim3 threadcount, blocks;

public:
    Convolve2D(ImgProperty input_prop_, Conv2DParams params_);
    ~Convolve2D();

    void set_kernel(const std::vector<FLOAT> &kernel_data);
    void forward(const FLOAT *input_device);

    FLOAT *get_output();
    Conv2DParams get_param();

    ImgProperty get_output_spec();
    ImgProperty get_input_spec();

    void validate_params();
};
