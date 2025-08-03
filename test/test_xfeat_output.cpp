#include "xfeat.hpp"
#include "tensorio.hpp"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <chrono>
#include <iomanip>
#include <numeric>
#include <fstream>

void print_tensor_info(const std::string& name, const DevicePointer<float>& tensor) {
    auto shape = tensor.get_shape();
    auto data = tensor.get_value();
    
    float min_val = *std::min_element(data.begin(), data.end());
    float max_val = *std::max_element(data.begin(), data.end());
    float mean_val = std::accumulate(data.begin(), data.end(), 0.0f) / data.size();
    
    std::cout << std::setw(15) << name << " | Shape: ";
    for (size_t i = 0; i < shape.size(); ++i) {
        std::cout << shape[i];
        if (i < shape.size() - 1) std::cout << "×";
    }
    std::cout << " | Range: [" << std::fixed << std::setprecision(4) 
              << min_val << ", " << max_val << "] | Mean: " << mean_val << std::endl;
}

int main() {
    try {
        std::cout << "=== XFeat C++/CUDA Implementation Test ===" << std::endl;
        
        // Load and validate image
        const std::string image_path = "../data/TajMahal.png";
        cv::Mat img = cv::imread(image_path, cv::IMREAD_GRAYSCALE);
        
        if (img.empty()) {
            std::cerr << "ERROR: Failed to load image from " << image_path << std::endl;
            return -1;
        }
        
        std::cout << "✓ Loaded image: " << img.cols << "×" << img.rows << std::endl;
        
        // Preprocess image
        cv::Mat img_float;
        img.convertTo(img_float, CV_32F, 1.0 / 255.0);
        
        std::vector<float> img_vec(img_float.begin<float>(), img_float.end<float>());
        std::vector<int> dims = {1, img.rows, img.cols};
        
        DevicePointer<float> img_device(img_vec, dims);
        
        // Initialize XFeat model
        const std::string model_path = "../params/xfeat_weights.h5";
        std::cout << "✓ Loading XFeat model from " << model_path << std::endl;
        
        auto start_time = std::chrono::high_resolution_clock::now();
        XFeat feat(model_path, img.rows, img.cols);
        auto init_time = std::chrono::high_resolution_clock::now();
        
        std::cout << "✓ Model initialized in " 
                  << std::chrono::duration_cast<std::chrono::milliseconds>(init_time - start_time).count() 
                  << "ms" << std::endl;
        
        // Run inference
        std::cout << "\n--- Running XFeat Inference ---" << std::endl;
        auto inference_start = std::chrono::high_resolution_clock::now();
        
        auto [heatmap, keypoints_folded, keypoints_, feats] = feat.forward(img_device);
        
        auto inference_end = std::chrono::high_resolution_clock::now();
        std::cout << "✓ Inference completed in " 
                  << std::chrono::duration_cast<std::chrono::milliseconds>(inference_end - inference_start).count() 
                  << "ms" << std::endl;
        
        // Print output statistics
        std::cout << "\n--- Output Statistics ---" << std::endl;
        print_tensor_info("Input", img_device);
        print_tensor_info("Heatmap", heatmap);
        print_tensor_info("Keypoints", keypoints_folded);
        print_tensor_info("Features", feats);
        
        // Create output directory
        tio::mkdir("./xfeat_output");
        
        // Save outputs to H5 format
        std::cout << "\n--- Saving Outputs ---" << std::endl;
        
        auto input_shape = img_device.get_shape();
        auto input_data = img_device.get_value();
        tio::save_hdf5(input_data, input_shape, "./xfeat_output/input.h5", "input");
        std::cout << "✓ Saved input.h5" << std::endl;
        
        auto heatmap_shape = heatmap.get_shape();
        auto heatmap_data = heatmap.get_value();
        tio::save_hdf5(heatmap_data, heatmap_shape, "./xfeat_output/heatmap.h5", "heatmap");
        std::cout << "✓ Saved heatmap.h5" << std::endl;
        
        auto keypoints_shape = keypoints_folded.get_shape();
        auto keypoints_data = keypoints_folded.get_value();
        tio::save_hdf5(keypoints_data, keypoints_shape, "./xfeat_output/keypoints.h5", "keypoints");
        std::cout << "✓ Saved keypoints.h5" << std::endl;
        
        auto feats_shape = feats.get_shape();
        auto feats_data = feats.get_value();
        tio::save_hdf5(feats_data, feats_shape, "./xfeat_output/features.h5", "features");
        std::cout << "✓ Saved features.h5" << std::endl;
        
        // Save metadata
        std::ofstream metadata("./xfeat_output/metadata.txt");
        metadata << "Image: " << image_path << std::endl;
        metadata << "Image size: " << img.cols << "×" << img.rows << std::endl;
        metadata << "Model: " << model_path << std::endl;
        metadata << "Inference time: " 
                 << std::chrono::duration_cast<std::chrono::milliseconds>(inference_end - inference_start).count() 
                 << "ms" << std::endl;
        metadata.close();
        
        // Visualize heatmap (optional)
        if (heatmap_shape.size() >= 2) {
            cv::Mat heatmap_img(heatmap_shape[heatmap_shape.size()-2], 
                               heatmap_shape[heatmap_shape.size()-1], 
                               CV_32F, heatmap_data.data());
            
            cv::Mat heatmap_vis;
            cv::normalize(heatmap_img, heatmap_vis, 0, 255, cv::NORM_MINMAX, CV_8U);
            cv::applyColorMap(heatmap_vis, heatmap_vis, cv::COLORMAP_JET);
            cv::imwrite("./xfeat_output/heatmap_visualization.png", heatmap_vis);
            std::cout << "✓ Saved heatmap visualization" << std::endl;
        }
        
        std::cout << "\n✅ XFeat test completed successfully!" << std::endl;
        std::cout << "📁 All outputs saved to ./xfeat_output/" << std::endl;
        
        return 0;
        
    } catch (const std::exception& e) {
        std::cerr << "❌ ERROR: " << e.what() << std::endl;
        return -1;
    }
}