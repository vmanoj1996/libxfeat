// test_pool.cpp
#include "pool.hpp"
#include <iostream>
#include <fstream>
#include <vector>
#include <opencv2/opencv.hpp>
#include <cstdlib>

#include "tensorio.hpp"

// Helper to save tensor data to a binary file
void save_tensor(const std::vector<FLOAT>& data, const std::string& filename) {
    std::ofstream file(filename, std::ios::binary);
    file.write(reinterpret_cast<const char*>(data.data()), data.size() * sizeof(FLOAT));
    file.close();
}

int main() {
    // Load image using OpenCV
    cv::Mat img = cv::imread("../data/TajMahal.png");
    if (img.empty()) {
        std::cerr << "Failed to load image ../data/TajMahal.png" << std::endl;
        return 1;
    }

    // Preprocess: BGR -> RGB, normalize to [0, 1], convert HWC -> CHW
    cv::Mat img_rgb;
    cv::cvtColor(img, img_rgb, cv::COLOR_BGR2RGB);
    cv::Mat img_float;
    img_rgb.convertTo(img_float, CV_32F, 1.0/255.0);

    int channels = img_float.channels();
    int height = img_float.rows;
    int width = img_float.cols;

    std::vector<FLOAT> host_input(channels * height * width);
    for (int h = 0; h < height; ++h) {
        for (int w = 0; w < width; ++w) {
            cv::Vec3f pixel = img_float.at<cv::Vec3f>(h, w);
            for (int c = 0; c < channels; ++c) {
                host_input[c * height * width + h * width + w] = pixel[c];
            }
        }
    }

    // --- Pooling Test ---
    const int pool_factor = 2;
    ImgProperty input_prop = {channels, height, width};

    // 1. Create a PoolParams object. A pool_factor of 2 usually means
    //    a kernel size of 2x2 and a stride of 2x2.
    PoolParams pool_params(pool_factor, pool_factor, pool_factor, pool_factor);

    // 2. Pass the pool_params object to the factory function.
    auto layer = avgpool2d(input_prop, pool_params);
    // -------------------

    DevicePointer<FLOAT> input(host_input, {channels, height, width});

    auto& output = layer->forward(input);
    std::vector<FLOAT> host_output = output.get_value();

    // Save results
    tio::mkdir("./test/pool");
    save_tensor(host_input,  "./test/pool/input.bin");
    save_tensor(host_output, "./test/pool/output.bin");

    // Save dimensions for Python verifier
    auto output_shape = output.get_shape();
    std::ofstream dims_file("./test/pool/dims.txt");
    dims_file << channels << " " << height << " " << width << " ";
    dims_file << output_shape[1] << " " << output_shape[2] << " " << pool_factor;
    dims_file.close();

    std::cout << "Saved average pooling test data to ./test/pool/" << std::endl;

    return 0;
}