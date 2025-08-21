/*
nsys profile --trace=nvtx ./perf_xfeat
nsight-sys report.nsys-rep 


*/

#include "xfeat.hpp"
#include "primitives.hpp" // Ensure this is included for DevicePointer

#include <iostream>
#include <chrono>
#include <vector>
#include <random>
#include <numeric>
#include <algorithm>
#include <iomanip> // For std::fixed and std::setprecision

#include <nvtx3/nvToolsExt.h>

// Include the CUDA runtime header for synchronization
#include <cuda_runtime.h>

int main() {
    std::cout << "Starting C++ performance test for XFeat forward pass..." << std::endl;
    
    // --- Configuration ---
    const int height = 480;
    const int width = 640;
    const int channels = 1;
    const int num_runs = 1000;
    const std::string model_path = "../params/xfeat_weights.h5";

    std::cout << "----------------------------------" << std::endl;
    std::cout << "Configuration:" << std::endl;
    std::cout << "  - Input Size: " << width << "x" << height << "x" << channels << std::endl;
    std::cout << "  - Model Path: " << model_path << std::endl;
    std::cout << "  - Number of Runs: " << num_runs << std::endl;
    std::cout << "----------------------------------" << std::endl;


    // --- Data Preparation ---
    std::cout << "Preparing input data on device..." << std::endl;
    // 1. Generate random host input data
    std::vector<float> host_input_data(channels * height * width);
    std::mt19937 rng(std::chrono::high_resolution_clock::now().time_since_epoch().count());
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);

    std::generate(host_input_data.begin(), host_input_data.end(), [&]() { return dist(rng); });

    // 2. Create DevicePointer using its default constructor
    DevicePointer<float> input; 
    // 3. Allocate device memory with the desired dimensions
    input.alloc({channels, height, width}); 
    // 4. Set the value from host data (this will copy to device)
    input.set_value(host_input_data);
    std::cout << "Input data prepared." << std::endl;

    // --- Model Initialization ---
    std::cout << "Initializing XFeat model..." << std::endl;
    XFeat model(model_path, height, width); 
    std::cout << "Model initialized." << std::endl;

    model.init(input);
    // --- Warm-up Run ---
    // A warm-up run is essential to ensure CUDA kernels are compiled,
    // caches are warm, and initial setup costs don't affect timing.
    std::cout << "Performing warm-up run..." << std::endl;
    try {
        model.forward(input);
        // Synchronize after the warm-up to ensure it's fully complete
        cudaDeviceSynchronize(); 
    } catch (const std::exception& e) {
        std::cerr << "Warm-up failed: " << e.what() << std::endl;
        return 1;
    }
    std::cout << "Warm-up complete." << std::endl;


    // --- Timed Performance Measurement ---
    std::cout << "Starting " << num_runs << " timed runs..." << std::endl;
    std::vector<double> timings_ms;
    timings_ms.reserve(num_runs);

    nvtxRangePush("XFEAT_forward");
    for (int i = 0; i < num_runs; ++i)
    {
        auto start_time = std::chrono::high_resolution_clock::now();

        model.forward(input);
        cudaDeviceSynchronize();

        auto end_time = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> elapsed_time = end_time - start_time;
        
        timings_ms.push_back(elapsed_time.count());
        // std::cout << "  Run " << std::setw(2) << i + 1 << "/" << num_runs << ": " << std::fixed << std::setprecision(3) << elapsed_time.count() << " ms" << std::endl;
    }
    nvtxRangePop();


    std::cout << "All runs complete." << std::endl;

    // --- Results Analysis ---
    if (timings_ms.empty()) {
        std::cerr << "No timing data was collected." << std::endl;
        return 1;
    }

    double total_time = std::accumulate(timings_ms.begin(), timings_ms.end(), 0.0);
    double average_time = total_time / timings_ms.size();

    // Calculate FPS from average latency in ms.
    // FPS = 1 second / average_time_in_seconds = 1000 ms / average_time_in_ms
    double average_fps = 1000.0 / average_time;

    // Sort timings to find min, max, and median
    std::sort(timings_ms.begin(), timings_ms.end());

    double min_time = timings_ms.front();
    double max_time = timings_ms.back();
    double median_time = timings_ms[timings_ms.size() / 2];

    std::cout << "\n--- Performance Summary ---" << std::endl;
    std::cout << std::fixed << std::setprecision(3);
    std::cout << "Total time for " << num_runs << " runs: " << total_time << " ms" << std::endl;
    std::cout << "Average latency:        " << average_time << " ms" << std::endl;
    std::cout << "Average throughput (FPS): " << average_fps << std::endl; // Added FPS output
    std::cout << "Median latency:         " << median_time << " ms" << std::endl;
    std::cout << "Minimum latency:        " << min_time << " ms" << std::endl;
    std::cout << "Maximum latency:        " << max_time << " ms" << std::endl;
    std::cout << "---------------------------\n" << std::endl;

    return 0;
}