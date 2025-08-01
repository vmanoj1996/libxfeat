#include <matx.h>
#include <chrono>
#include <iostream>
#include <vector>

using namespace matx;
using namespace std::chrono;

void benchmark_matx() {
    cudaExecutor exec{0};
    
    std::vector<std::pair<int, int>> sizes = {{1000, 1000}, {2000, 2000}, {4000, 4000}, {8000, 8000}};
    
    for (auto size : sizes) {
        int rows = size.first;
        int cols = size.second;
        
        std::cout << "\n🚀 MatX benchmark - Size: " << rows << "x" << cols << std::endl;
        
        // Create tensors using correct syntax
        auto a = make_tensor<float>({rows, cols});
        auto b = make_tensor<float>({rows, cols});
        auto c = make_tensor<float>({rows, cols});
        auto sum_result = make_tensor<float>({1});  // Scalar as 1-element tensor
        
        // Initialize with random values
        (a = random<float>({rows, cols}, NORMAL)).run(exec);
        (b = random<float>({rows, cols}, NORMAL)).run(exec);
        exec.sync();
        
        // Benchmark elementwise_add
        {
            // Warmup
            for (int i = 0; i < 10; i++) {
                (c = a + b).run(exec);
            }
            exec.sync();
            
            auto start = high_resolution_clock::now();
            for (int i = 0; i < 100; i++) {
                (c = a + b).run(exec);
            }
            exec.sync();
            auto end = high_resolution_clock::now();
            
            double avg_time = duration_cast<microseconds>(end - start).count() / 100.0 / 1000.0;
            std::cout << "  elementwise_add: " << std::fixed << std::setprecision(3) << avg_time << " ms" << std::endl;
        }
        
        // Benchmark elementwise_mul
        {
            for (int i = 0; i < 10; i++) {
                (c = a * b).run(exec);
            }
            exec.sync();
            
            auto start = high_resolution_clock::now();
            for (int i = 0; i < 100; i++) {
                (c = a * b).run(exec);
            }
            exec.sync();
            auto end = high_resolution_clock::now();
            
            double avg_time = duration_cast<microseconds>(end - start).count() / 100.0 / 1000.0;
            std::cout << "  elementwise_mul: " << std::fixed << std::setprecision(3) << avg_time << " ms" << std::endl;
        }
        
        // Benchmark matrix_mul
        {
            for (int i = 0; i < 10; i++) {
                (c = matmul(a, b)).run(exec);
            }
            exec.sync();
            
            auto start = high_resolution_clock::now();
            for (int i = 0; i < 100; i++) {
                (c = matmul(a, b)).run(exec);
            }
            exec.sync();
            auto end = high_resolution_clock::now();
            
            double avg_time = duration_cast<microseconds>(end - start).count() / 100.0 / 1000.0;
            std::cout << "  matrix_mul     : " << std::fixed << std::setprecision(3) << avg_time << " ms" << std::endl;
        }
        
        // Benchmark sin_exp
        {
            for (int i = 0; i < 10; i++) {
                (c = sin(a) + exp(b * 0.1f)).run(exec);
            }
            exec.sync();
            
            auto start = high_resolution_clock::now();
            for (int i = 0; i < 100; i++) {
                (c = sin(a) + exp(b * 0.1f)).run(exec);
            }
            exec.sync();
            auto end = high_resolution_clock::now();
            
            double avg_time = duration_cast<microseconds>(end - start).count() / 100.0 / 1000.0;
            std::cout << "  sin_exp        : " << std::fixed << std::setprecision(3) << avg_time << " ms" << std::endl;
        }
        
        // Benchmark reduction_sum
        {
            for (int i = 0; i < 10; i++) {
                (sum_result = sum(a + b)).run(exec);
            }
            exec.sync();
            
            auto start = high_resolution_clock::now();
            for (int i = 0; i < 100; i++) {
                (sum_result = sum(a + b)).run(exec);
            }
            exec.sync();
            auto end = high_resolution_clock::now();
            
            double avg_time = duration_cast<microseconds>(end - start).count() / 100.0 / 1000.0;
            std::cout << "  reduction_sum  : " << std::fixed << std::setprecision(3) << avg_time << " ms" << std::endl;
        }
    }
}

int main() {
    MATX_ENTER_HANDLER();
    
    std::cout << "MatX Performance Benchmark" << std::endl;
    std::cout << "=========================" << std::endl;
    
    try {
        benchmark_matx();
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    MATX_EXIT_HANDLER();
    return 0;
}