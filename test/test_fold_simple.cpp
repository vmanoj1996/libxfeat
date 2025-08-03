// test/test_fold.cpp
#include <cuda_runtime.h>
#include "fold.hpp"
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cmath>
#include <numeric>
#include <opencv2/opencv.hpp>
#include <cstdlib>
#include <matio.h>

// Helper to save tensor as .mat file
void save_tensor_mat(const std::vector<FLOAT>& data, int height, int width, const std::string& filename, const std::string& var_name) {
    mat_t* matfp = Mat_CreateVer(filename.c_str(), NULL, MAT_FT_MAT73);
    if (!matfp) {
        std::cerr << "Failed to create " << filename << std::endl;
        return;
    }
    
    size_t dims[2] = {static_cast<size_t>(height), static_cast<size_t>(width)};
    matvar_t* matvar = Mat_VarCreate(var_name.c_str(), MAT_C_SINGLE, MAT_T_SINGLE, 2, dims, (void*)data.data(), 0);
    Mat_VarWrite(matfp, matvar, MAT_COMPRESSION_NONE);
    
    Mat_VarFree(matvar);
    Mat_Close(matfp);
}

int main() {
    // --- 1. Create small sequential image ---
    
    const int height = 8;
    const int width = 8;
    
    std::vector<FLOAT> host_input(height * width);
    for (int i = 0; i < height * width; ++i) {
        host_input[i] = static_cast<FLOAT>(i + 1) / (height * width);
    }

    std::cout << "Created " << height << "x" << width << " sequential image" << std::endl;

    // --- 2. Instantiate Layers and Execute ---

    auto fold_layer = make_fold(height, width);
    auto unfold_layer = make_unfold(height, width);

    DevicePointer<FLOAT> input_d(host_input, {1, height, width});

    std::cout << "Performing Fold operation..." << std::endl;
    auto& folded_output_d = fold_layer->forward(input_d);

    auto folded_shape = folded_output_d.get_shape();
    DevicePointer<FLOAT> unfold_input_d;
    unfold_input_d.alloc({folded_shape[0] + 1, folded_shape[1], folded_shape[2]});

    size_t num_elements = std::accumulate(folded_shape.begin(), folded_shape.end(), 1, std::multiplies<size_t>());
    size_t size_in_bytes = num_elements * sizeof(FLOAT);

    cudaError_t err = cudaMemcpy(unfold_input_d.get(), folded_output_d.get(),
                                 size_in_bytes, cudaMemcpyDeviceToDevice);

    if (err != cudaSuccess) {
        std::cerr << "CUDA memcpy failed: " << cudaGetErrorString(err) << std::endl;
        return 1;
    }

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

    // --- 4. Save as .mat files ---

    system("mkdir -p ./fold_test");
    
    save_tensor_mat(host_input, height, width, "./fold_test/input.mat", "input");
    
    std::vector<FLOAT> host_folded_output = folded_output_d.get_value();
    save_tensor_mat(host_folded_output, folded_shape[1], folded_shape[2], "./fold_test/folded_output.mat", "folded");
    
    save_tensor_mat(host_output, height, width, "./fold_test/final_output.mat", "output");

    std::cout << "Saved .mat files to ./fold_test/" << std::endl;

    return (mean_error < tolerance) ? 0 : 1;
}