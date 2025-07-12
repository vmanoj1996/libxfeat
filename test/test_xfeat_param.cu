#include <matx.h>
#include <iostream>
#include "xfeat.hpp"

int main() {
    // Initialize CUDA
    matx::cudaExecutor exec{0};
    
    // Load XFeat parameters
    XFeatParams params;
    int result = params.loadParams("xfeat_weights.h5");
    
    if (result != 0) {
        std::cerr << "Failed to load parameters from xfeat_weights.h5" << std::endl;
        return 1;
    }
    
    std::cout << "Successfully loaded " << params.numParams() << " parameters!" << std::endl;
    
    // Print all parameters
    params.printParams();
    
    // Test fetching specific parameters with different ranks
    std::string param_4d = "net.block1.0.layer.0.weight";  // Likely 4D conv weight
    std::string param_1d = "net.block1.0.layer.0.bias";    // Likely 1D bias
    
    // Test 4D parameter
    if (params.hasParam(param_4d)) {
        auto shape = params.getShape(param_4d);
        std::cout << "\n=== Testing 4D Parameter: " << param_4d << " ===" << std::endl;
        std::cout << "Shape: [";
        for (size_t i = 0; i < shape.size(); ++i) {
            if (i > 0) std::cout << ", ";
            std::cout << shape[i];
        }
        std::cout << "]" << std::endl;
        
        if (shape.size() == 4) {
            // Get as 4D tensor
            auto weights_4d = params.makeParam4D(param_4d);
            std::cout << "Successfully created 4D tensor" << std::endl;
            std::cout << "First value: " << weights_4d(0, 0, 0, 0) << std::endl;
            
            // Test MatX operations on the weights - cast to matx::index_t
            auto scaled_weights = matx::make_tensor<float>({
                static_cast<matx::index_t>(shape[0]), 
                static_cast<matx::index_t>(shape[1]), 
                static_cast<matx::index_t>(shape[2]), 
                static_cast<matx::index_t>(shape[3])
            });
            (scaled_weights = weights_4d * 2.0f).run(exec);
            exec.sync();
            
            std::cout << "Scaled first value (x2): " << scaled_weights(0, 0, 0, 0) << std::endl;
        } else {
            std::cout << "Parameter is not 4D, has " << shape.size() << " dimensions" << std::endl;
        }
    } else {
        std::cout << "4D Parameter '" << param_4d << "' not found!" << std::endl;
    }
    
    // Test 1D parameter
    if (params.hasParam(param_1d)) {
        auto shape = params.getShape(param_1d);
        std::cout << "\n=== Testing 1D Parameter: " << param_1d << " ===" << std::endl;
        std::cout << "Shape: [";
        for (size_t i = 0; i < shape.size(); ++i) {
            if (i > 0) std::cout << ", ";
            std::cout << shape[i];
        }
        std::cout << "]" << std::endl;
        
        if (shape.size() == 1) {
            // Get as 1D tensor
            auto bias_1d = params.makeParam1D(param_1d);
            std::cout << "Successfully created 1D tensor" << std::endl;
            std::cout << "First value: " << bias_1d(0) << std::endl;
            
            // Test MatX operations - cast to matx::index_t
            auto scaled_bias = matx::make_tensor<float>({static_cast<matx::index_t>(shape[0])});
            (scaled_bias = bias_1d + 1.0f).run(exec);
            exec.sync();
            
            std::cout << "Offset first value (+1): " << scaled_bias(0) << std::endl;
        } else {
            std::cout << "Parameter is not 1D, has " << shape.size() << " dimensions" << std::endl;
        }
    } else {
        std::cout << "1D Parameter '" << param_1d << "' not found!" << std::endl;
    }
    
    // Test wrong rank access (should return empty tensor)
    std::cout << "\n=== Testing Wrong Rank Access ===" << std::endl;
    if (params.hasParam(param_4d)) {
        auto wrong_tensor = params.makeParam1D(param_4d);  // Try to get 4D param as 1D
        std::cout << "Trying to get 4D parameter as 1D tensor size: " << wrong_tensor.Size(0) << std::endl;
    }
    
    std::cout << "\nXFeatParams test completed successfully!" << std::endl;
    return 0;
}