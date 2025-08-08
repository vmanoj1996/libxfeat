// test/test_fold.cpp
#include <cuda_runtime.h>
#include "fold.hpp"
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cmath>
#include <numeric> // Required for std::accumulate
#include <opencv2/opencv.hpp>
#include <cstdlib>

// Helper to save tensor data to a binary file
void save_tensor(const std::vector<FLOAT>& data, const std::string& filename) {
    std::ofstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Failed to open " << filename << " for writing." << std::endl;
        return;
    }
    file.write(reinterpret_cast<const char*>(data.data()), data.size() * sizeof(FLOAT));
    file.close();
}

int main() {
    // --- 1. Setup and Preprocessing ---

    // Load image. The fold/unfold operation expects a single-channel (grayscale) image.
    cv::Mat img_bgr = cv::imread("../data/TajMahal.png");
    if (img_bgr.empty()) {
        std::cerr << "Failed to load image ../data/TajMahal.png" << std::endl;
        return 1;
    }

    cv::Mat img_gray;
    cv::cvtColor(img_bgr, img_gray, cv::COLOR_BGR2GRAY);

    // The fold/unfold kernels require dimensions to be a multiple of reduction_ratio (8).
    const int height = 256;
    const int width = 256;
    cv::Mat img_resized;
    cv::resize(img_gray, img_resized, cv::Size(width, height), 0, 0, cv::INTER_LINEAR);

    // Normalize to [0, 1] and convert to float
    cv::Mat img_float;
    img_resized.convertTo(img_float, CV_32F, 1.0 / 255.0);

    // Copy to a flat host vector
    std::vector<FLOAT> host_input(height * width);
    if (img_float.isContinuous()) {
        memcpy(host_input.data(), img_float.data, host_input.size() * sizeof(FLOAT));
    } else {
        std::cerr << "Matrix is not continuous, cannot perform direct memcpy." << std::endl;
        return 1;
    }

    // --- 2. Instantiate Layers and Execute ---

    // Create the layer instances using the helper factory functions from fold.hpp
    auto fold_layer = make_fold(height, width);
    auto unfold_layer = make_unfold(height, width);

    // Upload the input data to the device
    DevicePointer<FLOAT> input_d(host_input, {1, height, width});

    // Perform the Fold operation
    std::cout << "Performing Fold operation..." << std::endl;
    auto& folded_output_d = fold_layer->forward(input_d);

    // The Unfold layer expects 65 channels, but the Fold layer produces 64.
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

    // Perform the Unfold operation
    std::cout << "Performing Unfold operation..." << std::endl;
    auto& final_output_d = unfold_layer->forward(unfold_input_d);

    // --- 3. Verification and Cleanup ---

    // Retrieve the final result back to the host CPU
    std::cout << "Copying result back to host for verification..." << std::endl;
    std::vector<FLOAT> host_output = final_output_d.get_value();

    // Compare the final output with the original input
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
        std::cout << "Test FAILED: Reconstructed output does not match original input." << std::endl;
        std::cout << "(This is likely due to the suspected bug in the unfold_kernel indexing.)" << std::endl;
    }
    std::cout << "----------------------------------------" << std::endl;

    // --- 4. Save Artifacts for Debugging ---

    tio::mkdir("./test/fold_test");

    // Save binary tensors (for numerical verification)
    save_tensor(host_input, "./test/fold_test/input.bin");
    std::vector<FLOAT> host_folded_output = folded_output_d.get_value();
    save_tensor(host_folded_output, "./fold_test/folded_output.bin");
    save_tensor(host_output, "./test/fold_test/final_output.bin");

    // --- NEW CODE: SAVE IMAGES FOR VISUAL VERIFICATION ---
    
    // Convert the original preprocessed float image (range 0.0-1.0)
    // to a savable 8-bit image (range 0-255).
    cv::Mat input_img_to_save;
    img_float.convertTo(input_img_to_save, CV_8U, 255.0);
    cv::imwrite("./test/fold_test/input_image.png", input_img_to_save);

    // Wrap the final host_output vector in a cv::Mat header
    cv::Mat output_mat(height, width, CV_32F, host_output.data());
    // Convert the reconstructed float image to a savable 8-bit image
    cv::Mat output_img_to_save;
    output_mat.convertTo(output_img_to_save, CV_8U, 255.0);
    cv::imwrite("./test/fold_test/final_output_image.png", output_img_to_save);

    // --------------------------------------------------------

    // Save dimensions for Python verifier
    auto final_shape = final_output_d.get_shape();
    std::ofstream dims_file("./test/fold_test/dims.txt");
    dims_file << "input: " << input_d.get_shape()[0] << " " << input_d.get_shape()[1] << " " << input_d.get_shape()[2] << "\n";
    dims_file << "folded: " << folded_shape[0] << " " << folded_shape[1] << " " << folded_shape[2] << "\n";
    dims_file << "unfold_input: " << unfold_input_d.get_shape()[0] << " " << unfold_input_d.get_shape()[1] << " " << unfold_input_d.get_shape()[2] << "\n";
    dims_file << "final_output: " << final_shape[0] << " " << final_shape[1] << " " << final_shape[2] << "\n";
    dims_file.close();

    std::cout << "Saved fold/unfold test data and images to ./test/fold_test/" << std::endl;

    return (mean_error < tolerance) ? 0 : 1;
}
