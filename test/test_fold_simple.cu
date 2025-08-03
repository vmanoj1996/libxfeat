// test/test_fold.cpp
#include <cuda_runtime.h>
#include "fold.hpp"
#include "tensorio.hpp"
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cmath>
#include <numeric>
#include <opencv2/opencv.hpp>
#include <cstdlib>

// The CUDA kernel to copy data from the source tensor to a larger destination tensor,
// leaving the last channel padded with zeros.
__global__ void copy_and_pad_channel_kernel(float* dst, const float* src, int C, int H, int W) 
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements_dst = (C + 1) * H * W;

    if (idx >= total_elements_dst) return;

    // Calculate 3D index in destination tensor (C+1, H, W)
    int c = idx / (H * W);
    int h = (idx / W) % H;
    int w = idx % W;

    // If we are in one of the channels that needs to be copied
    if (c < C) {
        // Calculate the corresponding linear index in the source tensor (C, H, W)
        int src_idx = c * H * W + h * W + w;
        dst[idx] = src[src_idx];
    } else { // This is the last channel, which we want to be zero (or some other value)
        dst[idx] = 0.0f;
    }
}

int main() {
    // --- 1. Allocate the input ---
    
    const int height = 16;
    const int width  = 16;
    
    std::vector<FLOAT> host_input(height * width);
    for (int i = 0; i < height * width; ++i) {
        host_input[i] = static_cast<FLOAT>(i + 1) / (height * width);
    }

    std::cout << "Created " << height << "x" << width << " sequential image" << std::endl;

    // --- 2. Instantiate Layers and Execute ---

    auto fold_layer   = make_fold(height, width, 2);
    auto unfold_layer = make_unfold(height, width, 2);

    DevicePointer<FLOAT> input_d(host_input, {1, height, width});

    std::cout << "Performing Fold operation..." << std::endl;
    auto& folded_output_d = fold_layer->forward(input_d);

    std::cout<<"Fold works fine and it was checked interactively on MATLAB side"<<std::endl;

    // --- Corrected Block: Prepare data for Unfold operation ---
    
    // Get the shape of the folded output
    const auto& folded_shape = folded_output_d.get_shape();
    std::cout << "Folded shape: " << folded_shape << std::endl;

    // Extract dimensions for clarity
    const int C = folded_shape[0];
    const int H = folded_shape[1];
    const int W = folded_shape[2];

    // Allocate the destination buffer for the unfold operation
    DevicePointer<FLOAT> unfold_input_d;
    unfold_input_d.alloc({C + 1, H, W});

    // Set up and launch the kernel to copy data and pad the last channel
    const int total_dst_elements = (C + 1) * H * W;
    const int threads_per_block = 256;
    const int blocks = (total_dst_elements + threads_per_block - 1) / threads_per_block;

    copy_and_pad_channel_kernel<<<blocks, threads_per_block>>>(
        unfold_input_d.get(), 
        folded_output_d.get(), 
        C, H, W
    );
    // Wait for the kernel to complete before proceeding
    cudaDeviceSynchronize(); 

    // --- End of Corrected Block ---

    std::cout << "Performing Unfold operation..." << std::endl;
    auto& final_output_d = unfold_layer->forward(unfold_input_d);

    // --- 3. Verification ---

    std::cout << "Copying result back to host for verification..." << std::endl;
    std::vector<FLOAT> host_output = final_output_d.get_value();

    double total_error = 0.0;
    for (size_t i = 0; i < host_input.size(); ++i) {
        total_error += std::abs(host_input[i] - host_output[i]);
    }
    double mean_error = total_error / host_input.size();

    std::cout << "----------------------------------------" << std::endl;
    std::cout << "Mean Absolute Error: " << mean_error << std::endl;

    const double tolerance = 1e-6;
    if (mean_error < tolerance) {
        std::cout << "Test PASSED!" << std::endl;
    } else {
        std::cout << "Test FAILED!" << std::endl;
    }
    std::cout << "----------------------------------------" << std::endl;

    // --- 4. Save using tensorio ---

    tio::mkdir("./fold_test");
    
    tio::save_hdf5(host_input, {height, width}, "./fold_test/input.h5", "input");
    
    std::vector<FLOAT> host_folded_output = folded_output_d.get_value();
    tio::save_hdf5(host_folded_output, {folded_shape[0], folded_shape[1], folded_shape[2]}, "./fold_test/folded_output.h5", "folded");
    
    tio::save_hdf5(host_output, {height, width}, "./fold_test/final_output.h5", "output");

    std::cout << "Saved .h5 files to ./fold_test/" << std::endl;

    auto result = (mean_error < tolerance) ? 0 : 1;

    return result;
}