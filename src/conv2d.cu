// Simple convolve 2d (non batched)

/*
Requirements:
1. No batch needed
2. preserve the size of the input image
3. input is  C1*HxW
4. output is C2xHxW

5. Parameter count is c2xc1xk2xk1

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

__global__ void convolve2d_kernel(const FLOAT* input_device, const FLOAT* param_device, FLOAT *output_device, int height, int width, int k1, int k2, int c1, int c2)
{
    int gid_row = threadIdx.x + blockIdx.x * blockDim.x;
    int gid_col = threadIdx.y + blockIdx.y * blockDim.y;
    int gid_c2  = threadIdx.z + blockIdx.z * blockDim.z;

    if(gid_row<height && gid_col<width && gid_c2<c2)
    {
        FLOAT sum = 0.0f;
        
        for(int k=0; k<c1; k++)
        {
            for(int j=0; j<k2; j++)
            {
                for(int i = 0; i<k1; i++)
                {
                    int kernel_index = param_device[gid_c2*(c1*k2*k1) + k*(k2*k1) + j*(k1) + i];
                    


                    sum += input_device[i_index] * kernel_value;

                }
            }
        }

        int o_index = gid_c2*width*height + gid_row*width + gid_col;
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
    // no need to actually store this on cpu!
    // FLOAT param[c2][c1][k2][k1];

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
        blocks = dim3((height-1+TC)/TC, (width-1+TC)/TC, (c2-1+TC)/TC);

        cudaMalloc(&output_device, width*height*sizeof(FLOAT));
        cudaMemset(output_device, 0, width*height*sizeof(FLOAT));


        cudaMalloc(&param_device, k1*k2*c1*c2*sizeof(FLOAT));

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
        convolve2d_kernel<<<blocks, threadcount>>>(input_device, param_device, output_device, height, width, k1, k2, c1, c2);
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