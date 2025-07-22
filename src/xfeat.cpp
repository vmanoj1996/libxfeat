#include "xfeat.hpp"
#include "conv2d.hpp"
#include "fold.hpp"

XFeat::XFeat(std::string model_file): model_params(model_file)
{
    // Initialize the network architecture based on the documents
    // Architecture: {4, 8, 24, 64, 64, 128} channels with feature pyramid
    
    // Block 1: 1 -> 4 -> 8 -> 8 -> 24 channels
    block1_conv1 = std::make_unique<Conv2D>(
        ImgProperty(600, 800),  // assuming input size
        Conv2DParams(3, 3, 1, 4, 1, 1, 1, 1)  // 1->4 channels, 3x3 kernel, stride 1, pad 1
    );
    
    block1_conv2 = std::make_unique<Conv2D>(
        ImgProperty(600, 800),
        Conv2DParams(3, 3, 4, 8, 2, 2, 1, 1)  // 4->8 channels, stride 2 (downsample)
    );
    
    block1_conv3 = std::make_unique<Conv2D>(
        ImgProperty(300, 400),  // after stride 2
        Conv2DParams(3, 3, 8, 8, 1, 1, 1, 1)  // 8->8 channels
    );
    
    block1_conv4 = std::make_unique<Conv2D>(
        ImgProperty(300, 400),
        Conv2DParams(3, 3, 8, 24, 2, 2, 1, 1)  // 8->24 channels, stride 2
    );
    
    // Block 2: 24 -> 24 -> 24 channels
    block2_conv1 = std::make_unique<Conv2D>(
        ImgProperty(150, 200),
        Conv2DParams(3, 3, 24, 24, 1, 1, 1, 1)
    );
    
    block2_conv2 = std::make_unique<Conv2D>(
        ImgProperty(150, 200),
        Conv2DParams(3, 3, 24, 24, 1, 1, 1, 1)
    );
    
    // Block 3: 24 -> 64 -> 64 -> 64 channels
    block3_conv1 = std::make_unique<Conv2D>(
        ImgProperty(150, 200),
        Conv2DParams(3, 3, 24, 64, 2, 2, 1, 1)  // stride 2 downsample
    );
    
    block3_conv2 = std::make_unique<Conv2D>(
        ImgProperty(75, 100),
        Conv2DParams(3, 3, 64, 64, 1, 1, 1, 1)
    );
    
    block3_conv3 = std::make_unique<Conv2D>(
        ImgProperty(75, 100),
        Conv2DParams(1, 1, 64, 64, 1, 1, 0, 0)  // 1x1 conv
    );
    
    // Skip connection: avgpool + 1x1 conv (1 -> 24 channels)
    skip1_conv = std::make_unique<Conv2D>(
        ImgProperty(150, 200),  // after 4x4 avgpool
        Conv2DParams(1, 1, 1, 24, 1, 1, 0, 0)
    );
    
    // Fusion block for multi-scale features
    fusion_conv1 = std::make_unique<Conv2D>(
        ImgProperty(75, 100),
        Conv2DParams(3, 3, 64, 64, 1, 1, 1, 1)
    );
    
    fusion_conv2 = std::make_unique<Conv2D>(
        ImgProperty(75, 100),
        Conv2DParams(3, 3, 64, 64, 1, 1, 1, 1)
    );
    
    fusion_final = std::make_unique<Conv2D>(
        ImgProperty(75, 100),
        Conv2DParams(1, 1, 64, 64, 1, 1, 0, 0)
    );
    
    // Keypoint head - separate branch using fold/unfold
    keypoint_fold = std::make_unique<Fold2D>(600, 800);  // 8x8 folding
    
    keypoint_conv1 = std::make_unique<Conv2D>(
        ImgProperty(75, 100),  // after folding
        Conv2DParams(1, 1, 64, 64, 1, 1, 0, 0)
    );
    
    keypoint_final = std::make_unique<Conv2D>(
        ImgProperty(75, 100),
        Conv2DParams(1, 1, 64, 65, 1, 1, 0, 0)  // 64 positions + 1 dustbin
    );
    
    // Reliability head
    reliability_conv = std::make_unique<Conv2D>(
        ImgProperty(75, 100),
        Conv2DParams(1, 1, 64, 1, 1, 1, 0, 0)  // single channel output
    );
    
    // Load weights from HDF5
    loadWeights();
}

void XFeat::loadWeights()
{
    // Load weights for each layer from the HDF5 parameters
    
    // Block 1 weights
    auto block1_0_weight = model_params.getParam("net.block1.0.layer.0.weight");
    block1_conv1->set_kernel(block1_0_weight);
    
    auto block1_1_weight = model_params.getParam("net.block1.1.layer.0.weight");
    block1_conv2->set_kernel(block1_1_weight);
    
    auto block1_2_weight = model_params.getParam("net.block1.2.layer.0.weight");
    block1_conv3->set_kernel(block1_2_weight);
    
    auto block1_3_weight = model_params.getParam("net.block1.3.layer.0.weight");
    block1_conv4->set_kernel(block1_3_weight);
    
    // Block 2 weights
    auto block2_0_weight = model_params.getParam("net.block2.0.layer.0.weight");
    block2_conv1->set_kernel(block2_0_weight);
    
    auto block2_1_weight = model_params.getParam("net.block2.1.layer.0.weight");
    block2_conv2->set_kernel(block2_1_weight);
    
    // Skip connection
    auto skip1_weight = model_params.getParam("net.skip1.1.weight");
    skip1_conv->set_kernel(skip1_weight);
    
    // Add more weight loading for other layers...
    
    std::cout << "Weights loaded successfully!" << std::endl;
}

DevicePointer<FLOAT>& XFeat::forward(DevicePointer<FLOAT>& input)
{
    // Forward pass through the network
    // This is a simplified version - you'll need to add BatchNorm and ReLU operations
    
    // Block 1 forward
    auto& x = block1_conv1->forward(input, BatchNormRelu{/* params */});
    x = block1_conv2->forward(x, BatchNormRelu{/* params */});
    x = block1_conv3->forward(x, BatchNormRelu{/* params */});
    x = block1_conv4->forward(x, BatchNormRelu{/* params */});
    
    // Block 2 forward
    x = block2_conv1->forward(x, BatchNormRelu{/* params */});
    x = block2_conv2->forward(x, BatchNormRelu{/* params */});
    
    // Feature fusion and final outputs would go here...
    
    return x;  // Return descriptors
}