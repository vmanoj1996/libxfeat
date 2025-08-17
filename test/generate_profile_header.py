# generate_profile_configs.py

def read_layer_configs(filename):
    configs = []
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                parts = line.split()
                if len(parts) == 11:
                    k1, k2, ci, co, s1, s2, p1, p2, ich, ih, iw = map(int, parts)
                    configs.append((k1, k2, ci, co, s1, s2, p1, p2, ich, ih, iw))
    return configs

def generate_header(configs, output_file):
    # Remove duplicates while preserving order
    unique_configs = []
    seen = set()
    for config in configs:
        key = config[:8]  # k1, k2, ci, co, s1, s2, p1, p2
        if key not in seen:
            seen.add(key)
            unique_configs.append(config)
    
    with open(output_file, 'w') as f:
        f.write("// Auto-generated profile configurations\n")
        f.write("#pragma once\n\n")
        f.write("#include \"conv2d_gemm.hpp\"\n")
        f.write("#include <unordered_map>\n\n")
        
        # Generate one generic template function
        f.write("struct LayerConfig {int k1, k2, ci, co, s1, s2, p1, p2;int input_channels, input_height, input_width;};\n")
        f.write("template<int K1, int K2, int CI, int CO, int S1, int S2, int P1, int P2>\n")
        f.write("void profile_generic_config(const LayerConfig& config) {\n")
        f.write("    constexpr Conv2DParams params{K1, K2, CI, CO, S1, S2, P1, P2};\n")
        f.write("    \n")
        f.write("    ImgProperty input_prop{config.input_channels, config.input_height, config.input_width};\n")
        f.write("    \n")
        f.write("    size_t input_elements = config.input_channels * config.input_height * config.input_width;\n")
        f.write("    std::vector<FLOAT> host_input(input_elements, 1.0f);\n")
        f.write("    \n")
        f.write("    size_t kernel_elements = CO * CI * K1 * K2;\n")
        f.write("    std::vector<FLOAT> host_kernel(kernel_elements, 0.1f);\n")
        f.write("    \n")
        f.write("    auto conv_layer = conv2d<params>(input_prop, host_kernel, Identity(), 0);\n")
        f.write("    DevicePointer<FLOAT> input_device(host_input, {config.input_channels, config.input_height, config.input_width});\n")
        f.write("    \n")
        f.write("    auto& output_device = conv_layer->forward(input_device);\n")
        f.write("    \n")
        f.write("    std::cout << \"✓ Profiled config: \" << K1 << \"x\" << K2 << \" \" << CI << \"->\" << CO \n")
        f.write("              << \" stride=\" << S1 << \"x\" << S2 << \" pad=\" << P1 << \"x\" << P2 << std::endl;\n")
        f.write("}\n\n")
        
        # Generate hash function and dispatch map
        f.write("uint64_t config_hash(const LayerConfig& config) {\n")
        f.write("    return ((uint64_t)config.k1 << 56) | ((uint64_t)config.k2 << 48) |\n")
        f.write("           ((uint64_t)config.ci << 32) | ((uint64_t)config.co << 16) |\n") 
        f.write("           ((uint64_t)config.s1 << 12) | ((uint64_t)config.s2 << 8) |\n")
        f.write("           ((uint64_t)config.p1 << 4) | ((uint64_t)config.p2);\n")
        f.write("}\n\n")
        
        # Generate dispatch function
        f.write("void profile_layer_config(const LayerConfig& config) {\n")
        f.write("    static std::unordered_map<uint64_t, void(*)(const LayerConfig&)> dispatch_map = {\n")
        
        for i, config in enumerate(unique_configs):
            k1, k2, ci, co, s1, s2, p1, p2, ich, ih, iw = config
            hash_val = (k1 << 56) | (k2 << 48) | (ci << 32) | (co << 16) | (s1 << 12) | (s2 << 8) | (p1 << 4) | p2
            f.write(f"        {{{hash_val}ULL, profile_generic_config<{k1}, {k2}, {ci}, {co}, {s1}, {s2}, {p1}, {p2}>}}")
            if i < len(unique_configs) - 1:
                f.write(",")
            f.write("\n")
        
        f.write("    };\n\n")
        f.write("    uint64_t hash = config_hash(config);\n")
        f.write("    auto it = dispatch_map.find(hash);\n")
        f.write("    if (it != dispatch_map.end()) {\n")
        f.write("        it->second(config);\n")
        f.write("    } else {\n")
        f.write("        std::cout << \"⚠ Unsupported config: \" << config.k1 << \"x\" << config.k2 << \" \" \n")
        f.write("                  << config.ci << \"->\" << config.co << \" stride=\" << config.s1 << \"x\" << config.s2 \n")
        f.write("                  << \" pad=\" << config.p1 << \"x\" << config.p2 << std::endl;\n")
        f.write("    }\n")
        f.write("}\n")

if __name__ == "__main__":
    configs = read_layer_configs("../build/conv2d_layer_configs.txt")
    generate_header(configs, "../include/profile_configs.hpp")
    print(f"Generated profile_configs.hpp with {len(set(config[:8] for config in configs))} unique configurations")
    print(f"Total configs read: {len(configs)}")