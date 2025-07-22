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
    
    // Block 4: 64 -> 64 -> 64 -> 64 channels
    block4_conv1 = std::make_unique<Conv2D>(
        ImgProperty(75, 100),
        Conv2DParams(3, 3, 64, 64, 2, 2, 1, 1)  // stride 2 downsample to 1/16
    );
    
    block4_conv2 = std::make_unique<Conv2D>(
        ImgProperty(37, 50),  // 1/16 resolution
        Conv2DParams(3, 3, 64, 64, 1, 1, 1, 1)
    );
    
    block4_conv3 = std::make_unique<Conv2D>(
        ImgProperty(37, 50),
        Conv2DParams(3, 3, 64, 64, 1, 1, 1, 1)
    );
    
    // Block 5: 64 -> 128 -> 128 -> 128 -> 64 channels
    block5_conv1 = std::make_unique<Conv2D>(
        ImgProperty(37, 50),
        Conv2DParams(3, 3, 64, 128, 2, 2, 1, 1)  // stride 2 downsample to 1/32
    );
    
    block5_conv2 = std::make_unique<Conv2D>(
        ImgProperty(18, 25),  // 1/32 resolution
        Conv2DParams(3, 3, 128, 128, 1, 1, 1, 1)
    );
    
    block5_conv3 = std::make_unique<Conv2D>(
        ImgProperty(18, 25),
        Conv2DParams(3, 3, 128, 128, 1, 1, 1, 1)
    );
    
    block5_conv4 = std::make_unique<Conv2D>(
        ImgProperty(18, 25),
        Conv2DParams(1, 1, 128, 64, 1, 1, 0, 0)  // 1x1 conv back to 64 channels
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
    
    keypoint_conv2 = std::make_unique<Conv2D>(
        ImgProperty(75, 100),
        Conv2DParams(1, 1, 64, 64, 1, 1, 0, 0)
    );
    
    keypoint_conv3 = std::make_unique<Conv2D>(
        ImgProperty(75, 100),
        Conv2DParams(1, 1, 64, 64, 1, 1, 0, 0)
    );
    
    keypoint_final = std::make_unique<Conv2D>(
        ImgProperty(75, 100),
        Conv2DParams(1, 1, 64, 65, 1, 1, 0, 0)  // 64 positions + 1 dustbin
    );
    
    // Heatmap head (reliability)
    heatmap_conv1 = std::make_unique<Conv2D>(
        ImgProperty(75, 100),
        Conv2DParams(1, 1, 64, 64, 1, 1, 0, 0)
    );
    
    heatmap_conv2 = std::make_unique<Conv2D>(
        ImgProperty(75, 100),
        Conv2DParams(1, 1, 64, 64, 1, 1, 0, 0)
    );
    
    heatmap_final = std::make_unique<Conv2D>(
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
    
    // Block 3 weights
    auto block3_0_weight = model_params.getParam("net.block3.0.layer.0.weight");
    block3_conv1->set_kernel(block3_0_weight);
    
    auto block3_1_weight = model_params.getParam("net.block3.1.layer.0.weight");
    block3_conv2->set_kernel(block3_1_weight);
    
    auto block3_2_weight = model_params.getParam("net.block3.2.layer.0.weight");
    block3_conv3->set_kernel(block3_2_weight);
    
    // Skip connection
    auto skip1_weight = model_params.getParam("net.skip1.1.weight");
    skip1_conv->set_kernel(skip1_weight);
    
    // Fusion block weights
    auto fusion_0_weight = model_params.getParam("net.block_fusion.0.layer.0.weight");
    fusion_conv1->set_kernel(fusion_0_weight);
    
    auto fusion_1_weight = model_params.getParam("net.block_fusion.1.layer.0.weight");
    fusion_conv2->set_kernel(fusion_1_weight);
    
    auto fusion_2_weight = model_params.getParam("net.block_fusion.2.weight");
    fusion_final->set_kernel(fusion_2_weight);
    
    // Keypoint head weights
    // TODO: Implement keypoint head weight loading - need to check exact parameter names
    // auto keypoint_0_weight = model_params.getParam("net.keypoint_head.0.layer.0.weight");
    // keypoint_conv1->set_kernel(keypoint_0_weight);
    
    // auto keypoint_3_weight = model_params.getParam("net.keypoint_head.3.weight");
    // keypoint_final->set_kernel(keypoint_3_weight);
    
    // Reliability head weights  
    // TODO: Implement reliability head weight loading - need to check exact parameter names
    // auto reliability_weight = model_params.getParam("net.reliability_head.weight");
    // reliability_conv->set_kernel(reliability_weight);
    
    // TODO: Load BatchNorm parameters (running_mean, running_var, etc.) from buffers
    // TODO: Load bias parameters where applicable
    // TODO: Implement blocks 4 and 5 weight loading
    // TODO: Implement heatmap head weight loading
    
    std::cout << "Weights loaded successfully!" << std::endl;
}

DevicePointer<FLOAT>& XFeat::forward(DevicePointer<FLOAT>& input)
{
    // Forward pass through the network
    // This is a simplified version - you'll need to add BatchNorm and ReLU operations
    
    // Block 1 forward
    auto& x = block1_conv1->forward(input, BatchNormRelu{/* params */});
    auto& x1 = block1_conv2->forward(x, BatchNormRelu{/* params */});  // 1/2 resolution
    auto& x2 = block1_conv3->forward(x1, BatchNormRelu{/* params */});
    auto& x_block1 = block1_conv4->forward(x2, BatchNormRelu{/* params */});  // 1/4 resolution
    
    // Block 2 forward
    auto& x3 = block2_conv1->forward(x_block1, BatchNormRelu{/* params */});
    auto& x_block2 = block2_conv2->forward(x3, BatchNormRelu{/* params */});  // 1/4 resolution
    
    // Block 3 forward
    auto& x4 = block3_conv1->forward(x_block2, BatchNormRelu{/* params */});  // 1/8 resolution
    auto& x5 = block3_conv2->forward(x4, BatchNormRelu{/* params */});
    auto& x_block3 = block3_conv3->forward(x5, BatchNormRelu{/* params */});  // 1/8 resolution
    
    // Block 4 forward
    auto& x6 = block4_conv1->forward(x_block3, BatchNormRelu{/* params */});  // 1/16 resolution
    auto& x7 = block4_conv2->forward(x6, BatchNormRelu{/* params */});
    auto& x_block4 = block4_conv3->forward(x7, BatchNormRelu{/* params */});  // 1/16 resolution
    
    // Block 5 forward
    auto& x8 = block5_conv1->forward(x_block4, BatchNormRelu{/* params */});  // 1/32 resolution
    auto& x9 = block5_conv2->forward(x8, BatchNormRelu{/* params */});
    auto& x10 = block5_conv3->forward(x9, BatchNormRelu{/* params */});
    auto& x_block5 = block5_conv4->forward(x10, BatchNormRelu{/* params */});  // 1/32 resolution, 64 channels
    
    // Skip connection: AvgPool + 1x1 conv
    // TODO: Implement avgpool operation
    // auto& skip_pooled = avgpool_4x4(input);  // 1/4 resolution
    // auto& skip_features = skip1_conv->forward(skip_pooled, Identity{});
    
    // Feature fusion (multi-scale features)
    // TODO: Implement upsampling and element-wise addition
    // Merge features from {1/8, 1/16, 1/32} -> 1/8 resolution
    // auto& upsampled_block4 = upsample_2x(x_block4);  // 1/16 -> 1/8
    // auto& upsampled_block5 = upsample_4x(x_block5);  // 1/32 -> 1/8
    // auto& merged_features = element_wise_add(x_block3, upsampled_block4, upsampled_block5);
    
    // For now, just use block3 output as merged features
    auto& merged_features = x_block3;
    
    // Fusion block forward
    auto& fused1 = fusion_conv1->forward(merged_features, BatchNormRelu{/* params */});
    auto& fused2 = fusion_conv2->forward(fused1, BatchNormRelu{/* params */});
    auto& descriptors = fusion_final->forward(fused2, Identity{});  // Final descriptor output
    
    // Keypoint head (parallel branch)
    auto* folded_input = keypoint_fold->fold(input.get());  // 8x8 folding
    DevicePointer<FLOAT> folded_device(folded_input, {64, 75, 100});  // 64 channels from 8x8
    auto& kp1 = keypoint_conv1->forward(folded_device, BatchNormRelu{/* params */});
    auto& kp2 = keypoint_conv2->forward(kp1, BatchNormRelu{/* params */});
    auto& kp3 = keypoint_conv3->forward(kp2, BatchNormRelu{/* params */});
    auto& keypoints = keypoint_final->forward(kp3, Identity{});  // 65 channels (64 + dustbin)
    
    // Heatmap head (reliability from descriptors)
    auto& heat1 = heatmap_conv1->forward(descriptors, BatchNormRelu{/* params */});
    auto& heat2 = heatmap_conv2->forward(heat1, BatchNormRelu{/* params */});
    auto& heatmap = heatmap_final->forward(heat2, Identity{});  // Single channel
    // TODO: Apply sigmoid activation to heatmap
    
    // TODO: Return structured output with descriptors, keypoints, and heatmap
    // For now, just return descriptors
    return descriptors;
}