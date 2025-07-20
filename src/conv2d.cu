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
#include <stdexcept>
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

__global__ void convolve2d_kernel(const FLOAT *input_device, const FLOAT *kernel_device, FLOAT *output_device, Conv2DParams p, ImgProperty input_prop, ImgProperty output_prop)
{
    /* Parameter documentation:


    */
    int idx_co = threadIdx.x + blockIdx.x * blockDim.x; // channel output
    int out_row = threadIdx.y + blockIdx.y * blockDim.y;
    int out_col = threadIdx.z + blockIdx.z * blockDim.z;

    if (out_row < output_prop.height && out_col < output_prop.width && idx_co < p.co)
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

                    FLOAT input_value = ((in_row_start + kernel_row) >= 0 && (in_row_start + kernel_row) < input_prop.height &&
                                         (in_col_start + kernel_col) >= 0 && (in_col_start + kernel_col) < input_prop.width)
                                            ? input_device[idx_ci * input_prop.height * input_prop.width + (in_row_start + kernel_row) * input_prop.width + (in_col_start + kernel_col)]
                                            : 0.0f;

                    sum += input_value * kernel_value;
                }
            }
        }

        int o_index = idx_co * output_prop.height * output_prop.width + out_row * output_prop.width + out_col;
        output_device[o_index] = sum;
    }
}

class Convolve2D
{

private:
    FLOAT *kernel_device; // co, ci, k1, k2 order for cache optimality. Thats how pytorch is built too. optimized for row major operations
    FLOAT *output_device; // co x output_height x output_width

    Conv2DParams params;
    ImgProperty input_prop, output_prop;
    dim3 threadcount, blocks;

public:
    Convolve2D(ImgProperty input_prop_, Conv2DParams params_) 
        : input_prop(input_prop_), params(params_)
    {
        /*
        compute the params
        */

        output_prop.height = (input_prop.height + 2 * params.p1 - params.k1) / params.s1 + 1;
        output_prop.width  = (input_prop.width  + 2 * params.p2 - params.k2) / params.s2 + 1;

        /*
        Set the kernel launch configuration
        */
        const int TC = 8;
        threadcount = dim3(TC, TC, TC);
        blocks = dim3((params.co + TC - 1) / TC,
                      (output_prop.height + TC - 1) / TC,
                      (output_prop.width + TC - 1) / TC);

        /*
        Allocate the memory for output and kernel
        */
        cudaMalloc(&output_device,   params.co * output_prop.height * output_prop.width * sizeof(FLOAT));
        cudaMemset(output_device, 0, params.co * output_prop.height * output_prop.width * sizeof(FLOAT));
        cudaMalloc(&kernel_device,   params.co * params.ci * params.k1 * params.k2 * sizeof(FLOAT));
    
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
        convolve2d_kernel<<<blocks, threadcount>>>(input_device, kernel_device, output_device, params, input_prop, output_prop);
        cudaDeviceSynchronize();
    }

    void set_kernel(const std::vector<FLOAT>& kernel_data)
    {
        size_t expected_size = params.co * params.ci * params.k1 * params.k2;
        if (kernel_data.size() != expected_size) {
            printf("Error: Expected %zu weights, got %zu\n", expected_size, kernel_data.size());
            exit(1);
        }
        cudaMemcpy(kernel_device, kernel_data.data(), expected_size * sizeof(FLOAT), cudaMemcpyHostToDevice);
    }

    FLOAT *get_output()
    {
        return output_device;
    }

    Conv2DParams get_param()
    {
        return params;
    }

    void validate_params()
    {
        if (params.k1 % 2 == 0) {
            throw std::invalid_argument("k1 must be odd");
        }
        if (params.k2 % 2 == 0) {
            throw std::invalid_argument("k2 must be odd");
        }
        if (params.s1 <= 0) {
            throw std::invalid_argument("s1 (stride height) must be positive");
        }
        if (params.s2 <= 0) {
            throw std::invalid_argument("s2 (stride width) must be positive");
        }
        if (params.p1 < 0) {
            throw std::invalid_argument("p1 (padding height) must be non-negative");
        }
        if (params.p2 < 0) {
            throw std::invalid_argument("p2 (padding width) must be non-negative");
        }

    }
};

#ifdef ACTIVATE_MAIN
int main()
{
    ImgProperty input_prop = {40, 60};  // height, width
    Conv2DParams conv_params = {1, 1, 3, 16, 2, 2, 1, 1};  // k1,k2,ci,co,s1,s2,p1,p2
    
    Convolve2D convlayer(input_prop, conv_params); 

    float *input_device;
    cudaMalloc(&input_device, ci * height * width * sizeof(float));

    cudaMemset(input_device, 0, ci * height * width * sizeof(float));

    convlayer.forward(input_device);

    cudaFree(input_device);

    return 0;
}
#endif