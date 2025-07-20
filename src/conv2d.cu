// Simple convolve 2d (non batched)

/*
Requirements:
1. No batch needed
2. preserve the size of the input image
3. input is  ci*HxW
4. output is coxHxW
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

using FLOAT=float;

__global__ void convolve2d_kernel(const FLOAT* input_device, const FLOAT* param_device, FLOAT *output_device, int height, int width, int k1, int k2, int ci, int co)
{
    int idx_co  = threadIdx.x + blockIdx.x * blockDim.x;
    int idx_row = threadIdx.y + blockIdx.y * blockDim.y;
    int idx_col = threadIdx.z + blockIdx.z * blockDim.z;

    if(idx_row<height && idx_col<width && idx_co<co)
    {
        FLOAT sum = 0.0f;
        
        for(int idx_ci=0; idx_ci<ci; idx_ci++)
        {
            for(int kernel_row=-k1/2; kernel_row<=k1/2; kernel_row++)
            {
                int idx_kernel_row = kernel_row + k1/2;

                for(int kernel_col=-k2/2; kernel_col<=k2/2; kernel_col++)
                {
                    int idx_kernel_col = kernel_col + k2/2;

                    // co x ci x k1 x k2
                    FLOAT kernel_value = param_device[idx_co*(ci*k1*k2) + idx_ci*(k1*k2) + idx_kernel_row*(k2) + idx_kernel_col];

                    FLOAT input_value  =  ((idx_row + kernel_row)>=0 && (idx_row + kernel_row)<height && 
                                            (idx_col + kernel_col)>=0 && (idx_col + kernel_col)<width) ?
                                            input_device[idx_ci*height*width + (idx_row + kernel_row)*width + (idx_col + kernel_col)]:
                                            0.0f;

                    sum += input_value * input_value;
                }
            }
        }

        int o_index = idx_co*height*width + idx_row*width + idx_col;
        output_device[o_index] = sum;

        printf("%d ", o_index);

    }
}

template <int k1, int k2, int c1, int c2>
class Convolve2D
{
    static_assert(k1 % 2 == 1, "k1 must be odd");
    static_assert(k2 % 2 == 1, "k2 must be odd");

    private:
    FLOAT *param_device;
    // c2, c1, k2, k1 order for cache optimality

    // input is assumed to be allocated else where
    FLOAT *output_device;
    int width, height;

    const int TC = 8;
    dim3 threadcount = dim3(TC, TC, TC);
    dim3 blocks;

    public:
    Convolve2D(int height_, int width_): width(width_), height(height_)
    {
        blocks = dim3((c2-1+TC)/TC, (height-1+TC)/TC, (width-1+TC)/TC);

        cudaMalloc(&output_device, width*height*sizeof(FLOAT));
        cudaMemset(output_device, 0, width*height*sizeof(FLOAT));


        cudaMalloc(&param_device, k1*k2*ci*co*sizeof(FLOAT));

        set_param();
    }

    void set_param()
    {

    }

    ~Convolve2D()
    {
        if(output_device != nullptr) cudaFree(output_device);
    }

    void forward(const FLOAT *input_device)
    {
        convolve2d_kernel<<<blocks, threadcount>>>(input_device, param_device, output_device, height, width, k1, k2, ci, co);
        cudaDeviceSynchronize();
    }

};

 
int main()
{

    int width = 60;
    int height = 40;
    Convolve2D<1, 1, 2, 1> convlayer(height, width);

    float *input_device;
    cudaMalloc(&input_device, height*width*sizeof(float));

    cudaMemset(input_device, 0, height*width*sizeof(float));


    convlayer.forward(input_device);



    return 0;
}