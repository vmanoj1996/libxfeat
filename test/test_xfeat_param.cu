#include <matx.h>
#include <iostream>
#include "xfeat.hpp"

int main() {
    // Load XFeat parameters
    XFeatParams params;
    int result = params.loadParams("../params/xfeat_weights.h5");
    
    if (result != 0) {
        std::cerr << "Failed to load parameters" << std::endl;
        return 1;
    }
    
    std::cout << "Loaded " << params.numParams() << " parameters" << std::endl;
    params.printParams();
    
    auto weights = params.makeParam4D("net.block4.1.layer.0.weight");
    std::cout << "N D tensor first value: " << weights(0, 0, 0, 0) << std::endl;
    
    
    std::cout << "Test completed!" << std::endl;
    return 0;
}