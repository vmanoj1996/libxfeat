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
    
    {
        auto weights = params.makeParam0D("net.block_fusion.1.layer.1.num_batches_tracked");
        std::cout << "0 D tensor first value: " << weights() << std::endl;
    }

    {
        auto weights = params.makeParam1D("net.fine_matcher.1.running_var");
        std::cout << "1 D tensor first value: " << weights(0) << std::endl;
    }
    
    {
        auto weights = params.makeParam2D("net.fine_matcher.6.weight");
        std::cout << "2 D tensor first value: " << weights(0, 0) << std::endl;
    }
    
    {
        auto weights = params.makeParam4D("net.block4.1.layer.0.weight");
        std::cout << "4 D tensor first value: " << weights(0, 0, 0, 0) << std::endl;
    }
    
    std::cout << "Test completed!" << std::endl;
    return 0;
}