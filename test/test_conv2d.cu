// test/test_conv2d.cu

#include <iostream>
#include <vector>
#include <numeric>
#include <cstdlib>

#include "conv2d.hpp"
#include "tensorio.hpp" // For saving HDF5 files

int main() {
    // --- 1. Define Test Parameters ---
    std::cout << "--- Setting up Conv2D Test ---" << std::endl;
    const int height = 32;
    const int width = 32;
    const int in_channels = 3;
    const int out_channels = 8;
    const int kernel_size = 3;
    const int stride = 1;
    const int padding = 1;

    ImgProperty input_prop {in_channels, height, width};
    Conv2DParams params {
        kernel_size, kernel_size, // k1, k2
        in_channels, out_channels, // ci, co
        stride, stride,           // s1, s2
        padding, padding          // p1, p2
    };
    std::cout << "Input Properties: " << input_prop << std::endl;
    std::cout << "Conv2D Parameters: " << params << std::endl;


    // --- 2. Generate Synthetic Data ---
    std::cout << "\n--- Generating synthetic data ---" << std::endl;
    // Generate Input Data (C, H, W)
    size_t input_elements = in_channels * height * width;
    std::vector<FLOAT> host_input(input_elements);
    for (size_t i = 0; i < input_elements; ++i) {
        host_input[i] = static_cast<FLOAT>(i % 100) / 100.0f; // Simple repeating pattern
    }

    // Generate Kernel Data (Co, Ci, kH, kW) - PyTorch format
    size_t kernel_elements = out_channels * in_channels * kernel_size * kernel_size;
    std::vector<FLOAT> host_kernel(kernel_elements);
    for (size_t i = 0; i < kernel_elements; ++i) {
        host_kernel[i] = static_cast<FLOAT>((i + 1) % 50 - 25) / 50.0f; // Small-ish weights
    }
    std::cout << "Generated input tensor of size " << in_channels << "x" << height << "x" << width << std::endl;
    std::cout << "Generated kernel tensor of size " << out_channels << "x" << in_channels << "x" << kernel_size << "x" << kernel_size << std::endl;


    // --- 3. Instantiate Layer and Run Forward Pass ---
    std::cout << "\n--- Performing forward pass ---" << std::endl;
    auto conv_layer = conv2d(input_prop, params, host_kernel);
    
    DevicePointer<FLOAT> input_d(host_input, {in_channels, height, width});
    auto& output_d = conv_layer->forward(input_d);


    // --- 4. Get Output and Save All Tensors ---
    std::cout << "\n--- Saving tensors to HDF5 files ---" << std::endl;
    std::vector<FLOAT> host_output = output_d.get_value();
    const auto& output_shape = output_d.get_shape();

    system("mkdir -p ./conv_test");
    const std::string test_dir = "./conv_test/";

    // Save Input
    tio::save_hdf5(host_input, {in_channels, height, width}, test_dir + "input.h5", "input");

    // Save Kernel
    tio::save_hdf5(host_kernel, {out_channels, in_channels, kernel_size, kernel_size}, test_dir + "kernel.h5", "kernel");

    // Save C++ Output
    tio::save_hdf5(host_output, {output_shape[0], output_shape[1], output_shape[2]}, test_dir + "output.h5", "output");
    
    std::cout << "Saved input.h5, kernel.h5, and output.h5 to " << test_dir << std::endl;
    std::cout << "Verification can now be run with the Python script." << std::endl;
    
    return 0;
}