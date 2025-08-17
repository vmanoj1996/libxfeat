// Auto-generated profile configurations
#pragma once

#include "conv2d_gemm.hpp"
#include <unordered_map>

struct LayerConfig {int k1, k2, ci, co, s1, s2, p1, p2;int input_channels, input_height, input_width;};
template<int K1, int K2, int CI, int CO, int S1, int S2, int P1, int P2>
void profile_generic_config(const LayerConfig& config) {
    constexpr Conv2DParams params{K1, K2, CI, CO, S1, S2, P1, P2};
    
    ImgProperty input_prop{config.input_channels, config.input_height, config.input_width};
    
    size_t input_elements = config.input_channels * config.input_height * config.input_width;
    std::vector<FLOAT> host_input(input_elements, 1.0f);
    
    size_t kernel_elements = CO * CI * K1 * K2;
    std::vector<FLOAT> host_kernel(kernel_elements, 0.1f);
    
    auto conv_layer = conv2d<params>(input_prop, host_kernel, Identity(), 0);
    DevicePointer<FLOAT> input_device(host_input, {config.input_channels, config.input_height, config.input_width});
    
    auto& output_device = conv_layer->forward(input_device);
    
    std::cout << "✓ Profiled config: " << K1 << "x" << K2 << " " << CI << "->" << CO 
              << " stride=" << S1 << "x" << S2 << " pad=" << P1 << "x" << P2 << std::endl;
}

uint64_t config_hash(const LayerConfig& config) {
    return ((uint64_t)config.k1 << 56) | ((uint64_t)config.k2 << 48) |
           ((uint64_t)config.ci << 32) | ((uint64_t)config.co << 16) |
           ((uint64_t)config.s1 << 12) | ((uint64_t)config.s2 << 8) |
           ((uint64_t)config.p1 << 4) | ((uint64_t)config.p2);
}

void profile_layer_config(const LayerConfig& config) {
    static std::unordered_map<uint64_t, void(*)(const LayerConfig&)> dispatch_map = {
        {217017211339149585ULL, profile_generic_config<3, 3, 1, 4, 1, 1, 1, 1>},
        {217017224224317969ULL, profile_generic_config<3, 3, 4, 8, 2, 2, 1, 1>},
        {217017241404182801ULL, profile_generic_config<3, 3, 8, 8, 1, 1, 1, 1>},
        {217017241405235729ULL, profile_generic_config<3, 3, 8, 24, 2, 2, 1, 1>},
        {72339073311183104ULL, profile_generic_config<1, 1, 1, 24, 1, 1, 0, 0>},
        {217017310124708113ULL, profile_generic_config<3, 3, 24, 24, 1, 1, 1, 1>},
        {217017310127333905ULL, profile_generic_config<3, 3, 24, 64, 2, 2, 1, 1>},
        {217017481926021393ULL, profile_generic_config<3, 3, 64, 64, 1, 1, 1, 1>},
        {72339343896744192ULL, profile_generic_config<1, 1, 64, 64, 1, 1, 0, 0>},
        {217017481926025745ULL, profile_generic_config<3, 3, 64, 64, 2, 2, 1, 1>},
        {217017481930220049ULL, profile_generic_config<3, 3, 64, 128, 2, 2, 1, 1>},
        {217017756808122641ULL, profile_generic_config<3, 3, 128, 128, 1, 1, 1, 1>},
        {72339618774651136ULL, profile_generic_config<1, 1, 128, 64, 1, 1, 0, 0>},
        {72339343892615424ULL, profile_generic_config<1, 1, 64, 1, 1, 1, 0, 0>},
        {72339343896809728ULL, profile_generic_config<1, 1, 64, 65, 1, 1, 0, 0>}
    };

    uint64_t hash = config_hash(config);
    auto it = dispatch_map.find(hash);
    if (it != dispatch_map.end()) {
        it->second(config);
    } else {
        std::cout << "⚠ Unsupported config: " << config.k1 << "x" << config.k2 << " " 
                  << config.ci << "->" << config.co << " stride=" << config.s1 << "x" << config.s2 
                  << " pad=" << config.p1 << "x" << config.p2 << std::endl;
    }
}
