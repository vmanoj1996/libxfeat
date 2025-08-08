#include "activation.hpp"
#include <iostream>
#include <fstream>
#include <vector>
#include <cstdlib>
#include "tensorio.hpp"

void save_tensor(const std::vector<FLOAT>& data, const std::string& filename) 
{
    std::ofstream file(filename, std::ios::binary);
    file.write(reinterpret_cast<const char*>(data.data()), data.size() * sizeof(FLOAT));
    file.close();
}

int main() 
{
    ImgProperty prop = {3, 4, 5};
    int total_size = prop.channels * prop.height * prop.width;
    
    // Create layer
    auto layer = activation(prop, Sigmoid(), 0);
    
    // Create and initialize input
    std::vector<FLOAT> host_input(total_size);
    for (int i = 0; i < total_size; i++) {
        host_input[i] = static_cast<FLOAT>(i - total_size/2) * 0.1f;
    }
    
    DevicePointer<FLOAT> input(host_input, {prop.channels, prop.height, prop.width});
    
    // Forward pass
    auto& output = layer->forward(input);
    
    // Get output data
    std::vector<FLOAT> host_output = output.get_value();
    
    // Save results
    tio::mkdir("./test/sigmoid");
    save_tensor(host_input, "./test/sigmoid/input.bin");
    save_tensor(host_output, "./test/sigmoid/output.bin");
    
    std::cout << "Saved " << total_size << " elements to ./test/sigmoid/" << std::endl;
    return 0;
}