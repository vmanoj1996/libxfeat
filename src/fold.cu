/*
Fold and unfold operations for xfeat network
input 1xHxW
output 64x(H/8)x(w/8)

and vice versa

nvcc -std=c++20 -arch=sm_89 fold.cu && ./a.out
*/

#include <cuda_runtime.h>
#include <cassert>
#include <cstdio>
#include <iostream>

#define FLOAT float

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

__global__ void unfold_kernel(const FLOAT *input_device, FLOAT *output_device, int height, int width, int ratio)
{
    // output idx
    int idx1 = threadIdx.x + blockIdx.x * blockDim.x;
    int idx2 = threadIdx.y + blockIdx.y * blockDim.y;

    if (idx1 < height && idx2 < width)
    {
        // compute the input dimension and index corresponding to idx1, idx2 inputs
        int idx0_in = (idx1 % ratio) * ratio + (idx2 % ratio);
        int idx1_in = idx1 / ratio;
        int idx2_in = idx2 / ratio;

        // int idx0_dim = ratio*ratio;
        int idx1_dim = height / ratio;
        int idx2_dim = width / ratio;

        output_device[idx1 * width + idx2] = input_device[idx0_in * (idx1_dim * idx2_dim) + idx1_in * idx2_dim + idx2_in];
    }
}

class Fold2D
{
private:
    FLOAT *output_device; // can mean folded or unfolded. This class does not care what exactly is stored in the output side
    int height, width;
    const int reduction_ratio = 8;
    int channel_out = reduction_ratio * reduction_ratio;

public:
    Fold2D(int height_, int width_) : height(height_), width(width_)
    {
        if (height % reduction_ratio != 0 || width % reduction_ratio != 0)
        {
            std::cerr << "height and width should be multiple of " << reduction_ratio << std::endl;
            exit(1);
        }
        cudaMallocManaged(&output_device, width * height * sizeof(FLOAT));
    }

    ~Fold2D()
    {
        if (output_device)
        {
            cudaFree(output_device);
        }
    }

    FLOAT *fold(const FLOAT *input_device)
    {
        if (input_device == output_device)
        {
            std::cerr << "input cannot be equal to output pointer in fold\n";
            exit(1);
        }

        const int TC = 16;
        dim3 threads(TC, TC);
        dim3 blocks((height + TC - 1) / TC, (width + TC - 1) / TC);

        fold_kernel<<<blocks, threads>>>(input_device, output_device, height, width, reduction_ratio);

        cudaDeviceSynchronize();

        return output_device;
    }

    FLOAT *unfold(const FLOAT *input_device)
    {
        if (input_device == output_device)
        {
            std::cerr << "input cannot be equal to output pointer in unfold\n";
            exit(1);
        }
        const int TC = 16;
        dim3 threads(TC, TC);
        dim3 blocks((height + TC - 1) / TC, (width + TC - 1) / TC);

        unfold_kernel<<<blocks, threads>>>(input_device, output_device, height, width, reduction_ratio);

        cudaDeviceSynchronize();

        return output_device;
    }

    FLOAT *get_output()
    {
        return output_device;
    }
};

int main()
{
    const int H = 16, W = 16;
    const int size = H * W;

    // Managed memory
    FLOAT *data, *data_output;
    cudaMallocManaged(&data, size * sizeof(FLOAT));
    cudaMallocManaged(&data_output, size * sizeof(FLOAT));

    // Initialize with numbers 1-256
    for (int i = 0; i < size; i++)
    {
        data[i] = i + 1;
    }

    // Test fold/unfold
    Fold2D folder(H, W);
    Fold2D unfolder(H, W);

    FLOAT *out_folded = folder.fold(data);
    FLOAT *out_unfolded = unfolder.unfold(out_folded);

    // Verify
    cudaDeviceSynchronize();
    bool success = true;
    for (int i = 0; i < size; i++)
    {
        printf("%d ", (int)data[i]);
    }
    
    printf("\n\n");
    for (int i = 0; i < 64; i++)
    {
        for (int j = 1; j < 2; j++)
        {
            for (int k = 0; k < 1; k++)
            {
                printf("%d ", (int)out_folded[i * 4 + j * 2 + k]);
            }
        }
    }
    printf("\n\n");
    for (int i = 0; i < size; i++)
    {
        printf("%d ", (int)out_unfolded[i]);
    }


    printf("\n");

    printf("Test %s\n", success ? "PASSED" : "FAILED");

    cudaFree(data);
    return 0;
}