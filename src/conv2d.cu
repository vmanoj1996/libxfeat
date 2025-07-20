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

__global__ void convolve2d_kernel(const FLOAT* input_device, const FLOAT* param_device, FLOAT *output_device, 
                                    int input_height, int input_width,
                                    int output_height, int output_width, 
                                    int k1, int k2, int ci, int co, 
                                    int s1, int s2, int p1, int p2)
{
    /* Parameter documentation:


    */
    int idx_co  = threadIdx.x + blockIdx.x * blockDim.x; // channel output 
    int out_row = threadIdx.y + blockIdx.y * blockDim.y;
    int out_col = threadIdx.z + blockIdx.z * blockDim.z;

    if(out_row<output_height && out_col<output_width && idx_co<co)
    {
        FLOAT sum = 0.0f;

        // once padded, the first operation that will happen is on this particular index in the imaginary padded input (implicit)
        int in_row_start = out_row * s1 - p1;
        int in_col_start = out_col * s2 - p2;
        
        for(int idx_ci=0; idx_ci<ci; idx_ci++)
        {
            for(int kernel_row=0; kernel_row<k1; kernel_row++)
            {
                for(int kernel_col=0; kernel_col<k2; kernel_col++)
                {
                    // co x ci x k1 x k2
                    FLOAT kernel_value = param_device[idx_co*(ci*k1*k2) + idx_ci*(k1*k2) + kernel_row*(k2) + kernel_col];

                    FLOAT input_value  =  ((in_row_start + kernel_row)>=0 && (in_row_start + kernel_row)<input_height && 
                                            (in_col_start + kernel_col)>=0 && (in_col_start + kernel_col)<input_width) ?
                                            input_device[idx_ci*input_height*input_width + (in_row_start + kernel_row)*input_width + (in_col_start + kernel_col)]:
                                            0.0f;

                    sum += input_value * kernel_value;
                }
            }
        }

        int o_index = idx_co*output_height*output_width + out_row*output_width + out_col;
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