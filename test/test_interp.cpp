// test_interp.cpp
#include "interp.hpp"
#include <iostream>
#include <fstream>
#include <vector>
#include <opencv2/opencv.hpp>
#include <cstdlib>
#include "tensorio.hpp"

void save_tensor(const std::vector<FLOAT>& data, const std::string& filename) 
{
    std::ofstream file(filename, std::ios::binary);
    file.write(reinterpret_cast<const char*>(data.data()), data.size() * sizeof(FLOAT));
    file.close();
}

int main() 
{
    // Load image using OpenCV
    cv::Mat img = cv::imread("../data/TajMahal.png");
    if (img.empty()) {
        std::cerr << "Failed to load image ../data/TajMahal.png" << std::endl;
        return 1;
    }
    
    // Convert BGR to RGB and normalize to [0, 1]
    cv::Mat img_rgb;
    cv::cvtColor(img, img_rgb, cv::COLOR_BGR2RGB);
    cv::Mat img_float;
    img_rgb.convertTo(img_float, CV_32F, 1.0/255.0);
    
    // Setup dimensions
    int channels = 3;
    int height = img_float.rows;
    int width = img_float.cols;
    ImgProperty input_prop = {channels, height, width};
    
    // Target size (4x upsampling)
    int target_height = height * 4;
    int target_width = width * 4;
    
    // Convert to CHW format for GPU
    std::vector<FLOAT> host_input(channels * height * width);
    for (int h = 0; h < height; h++) {
        for (int w = 0; w < width; w++) {
            cv::Vec3f pixel = img_float.at<cv::Vec3f>(h, w);
            for (int c = 0; c < channels; c++) {
                host_input[c * height * width + h * width + w] = pixel[c];
            }
        }
    }
    
    // Create layer
    auto layer = interp2d(input_prop, target_height, target_width, 0);
    
    DevicePointer<FLOAT> input(host_input, {channels, height, width});
    
    // Forward pass
    auto& output = layer->forward(input);
    
    // Get output data
    std::vector<FLOAT> host_output = output.get_value();
    
    // Save results
    tio::mkdir("./test/interp");
    
    save_tensor(host_input, "./test/interp/input.bin");
    save_tensor(host_output, "./test/interp/output.bin");
    
    // Save dimensions for Python
    std::ofstream dims_file("./test/interp/dims.txt");
    dims_file << channels << " " << height << " " << width << " ";
    dims_file << target_height << " " << target_width;
    dims_file.close();
    
    std::cout << "Saved interpolation test: " << height << "x" << width;
    
    return 0;
}