#include <matx.h>
#include <iostream>

int main() {
    // Initialize CUDA
    matx::cudaExecutor exec{0};
    
    // Create simple tensors
    auto t1 = matx::make_tensor<float>({10});
    auto t2 = matx::make_tensor<float>({10});
    auto t3 = matx::make_tensor<float>({10});
    
    // Fill with values
    (t1 = 1.0f).run(exec);
    (t2 = 2.0f).run(exec);
    
    // Simple addition
    (t3 = t1 + t2).run(exec);
    
    exec.sync();
    
    // Print result
    std::cout << "MatX installation test passed!" << std::endl;
    std::cout << "Result: " << t3(0) << " (should be 3.0)" << std::endl;
    
    return 0;
}