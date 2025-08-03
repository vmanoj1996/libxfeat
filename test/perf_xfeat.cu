#include "xfeat.hpp"
#include <iostream>
#include <chrono>

int main() {
    std::cout << "Running performance test..." << std::endl;
    
    const int height = 480;
    const int width = 640;
    const int num_runs = 50;

    XFeat model("path/to/your/model.h5", height, width);
    DevicePointer<float> input({1, height, width});
    // Maybe fill input with some data...

    // Warm-up run
    model.forward(input);

    auto start_time = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < num_runs; ++i) {
        model.forward(input);
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> total_time = end_time - start_time;
    double average_time = total_time.count() / num_runs;

    std::cout << "Performance test finished." << std::endl;
    std::cout << "Average forward pass time: " << average_time << " ms" << std::endl;

    return 0; // Success
}