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

// how to increase the stack size if needed
// g++ -Wl,--stack,16777216 program.cpp -o program
// g++ -fsanitize=address -g program.cpp -o program
// g++ -fsanitize=address,undefined,leak -g program.cpp

// nvcc -std=c++20 -arch=sm_89 conv2d.cu && ./a.out

#include <cuda_runtime.h>
#include <cassert>
#include <cstdio>

using FLOAT = float;

struct Conv2DParams
{
    int input_height, input_width;
    int output_height, output_width;
    int k1, k2, ci, co;
    int s1, s2, p1, p2;
};

__global__ void convolve2d_kernel(const FLOAT *input_device, const FLOAT *kernel_device, FLOAT *output_device, Conv2DParams p)
{
    /* Parameter documentation:


    */
    int idx_co = threadIdx.x + blockIdx.x * blockDim.x; // channel output
    int out_row = threadIdx.y + blockIdx.y * blockDim.y;
    int out_col = threadIdx.z + blockIdx.z * blockDim.z;

    if (out_row < p.output_height && out_col < p.output_width && idx_co < p.co)
    {
        FLOAT sum = 0.0f;

        // once padded, the first operation that will happen is on this particular index in the imaginary padded input (implicit)
        int in_row_start = out_row * p.s1 - p.p1;
        int in_col_start = out_col * p.s2 - p.p2;

        for (int idx_ci = 0; idx_ci < p.ci; idx_ci++)
        {
            for (int kernel_row = 0; kernel_row < p.k1; kernel_row++)
            {
                for (int kernel_col = 0; kernel_col < p.k2; kernel_col++)
                {
                    // co x ci x k1 x k2
                    FLOAT kernel_value = kernel_device[idx_co * (p.ci * p.k1 * p.k2) + idx_ci * (p.k1 * p.k2) + kernel_row * (p.k2) + kernel_col];

                    FLOAT input_value = ((in_row_start + kernel_row) >= 0 && (in_row_start + kernel_row) < p.input_height &&
                                         (in_col_start + kernel_col) >= 0 && (in_col_start + kernel_col) < p.input_width)
                                            ? input_device[idx_ci * p.input_height * p.input_width + (in_row_start + kernel_row) * p.input_width + (in_col_start + kernel_col)]
                                            : 0.0f;

                    sum += input_value * kernel_value;
                }
            }
        }

        int o_index = idx_co * p.output_height * p.output_width + out_row * p.output_width + out_col;
        output_device[o_index] = sum;
    }
}

template <int k1, int k2, int s1, int s2, int p1, int p2>
class Convolve2D
{
    static_assert(k1 % 2 == 1, "k1 must be odd");
    static_assert(k2 % 2 == 1, "k2 must be odd");

private:
    FLOAT *kernel_device; // co, ci, k1, k2 order for cache optimality. Thats how pytorch is built too. optimized for row major operations
    FLOAT *output_device; // co x output_height x output_width

    Conv2DParams params;
    dim3 threadcount, blocks;

public:
    Convolve2D(int input_height_, int input_width_, int ci_, int co_)
    {
        /*
        compute the params
        */

        params.input_height = input_height_;
        params.input_width = input_width_;
        params.ci = ci_;
        params.co = co_;
        params.k1 = k1;
        params.k2 = k2;
        params.s1 = s1;
        params.s2 = s2;
        params.p1 = p1;
        params.p2 = p2;

        params.output_height = (input_height_ + 2 * p1 - k1) / s1 + 1;
        params.output_width = (input_width_ + 2 * p2 - k2) / s2 + 1;

        /*
        Set the kernel launch configuration
        */
        const int TC = 8;
        threadcount = dim3(TC, TC, TC);
        blocks = dim3((params.co + TC - 1) / TC,
                      (params.output_height + TC - 1) / TC,
                      (params.output_width + TC - 1) / TC);

        /*
        Allocate the memory for output and kernel
        */
        cudaMalloc(&output_device, params.co * params.output_height * params.output_width * sizeof(FLOAT));
        cudaMemset(output_device, 0, params.co * params.output_height * params.output_width * sizeof(FLOAT));
        cudaMalloc(&kernel_device, params.co * params.ci * k1 * k2 * sizeof(FLOAT));
    }

    ~Convolve2D()
    {
        if (output_device != nullptr)
            cudaFree(output_device);
        if (kernel_device != nullptr)
            cudaFree(kernel_device);
    }

    void forward(const FLOAT *input_device)
    {
        convolve2d_kernel<<<blocks, threadcount>>>(input_device, kernel_device, output_device, params);
        cudaDeviceSynchronize();
    }

    FLOAT *get_output()
    {
        return output_device;
    }

    Conv2DParams get_param()
    {
        return params;
    }
};

#ifdef ACTIVATE_MAIN
int main()
{
    int width = 60;
    int height = 40;
    int ci = 3;
    int co = 16;

    Convolve2D<1, 1, 2, 2, 1, 1> convlayer(height, width, ci, co);

    float *input_device;
    cudaMalloc(&input_device, ci * height * width * sizeof(float));

    cudaMemset(input_device, 0, ci * height * width * sizeof(float));

    convlayer.forward(input_device);

    cudaFree(input_device);

    return 0;
}
#endif