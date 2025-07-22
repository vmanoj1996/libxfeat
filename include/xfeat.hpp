#pragma once
#include "xfeat_params.hpp"

#include "conv2d.hpp"
#include "fold.hpp"

#include <memory>

class XFeat
{
private:
    XFeatParams model_params;
    
    // Network layers
    std::unique_ptr<Conv2D> block1_conv1, block1_conv2, block1_conv3, block1_conv4;
    std::unique_ptr<Conv2D> block2_conv1, block2_conv2;
    std::unique_ptr<Conv2D> block3_conv1, block3_conv2, block3_conv3;
    std::unique_ptr<Conv2D> skip1_conv;
    std::unique_ptr<Conv2D> fusion_conv1, fusion_conv2, fusion_final;
    std::unique_ptr<Conv2D> keypoint_conv1, keypoint_final;
    std::unique_ptr<Conv2D> reliability_conv;
    
    std::unique_ptr<Fold2D> keypoint_fold;
    
    void loadWeights();
    
public:
    XFeat(std::string model_file);
    ~XFeat() = default;
    
    DevicePointer<FLOAT>& forward(DevicePointer<FLOAT>& input);
    
    // Disable copy operations
    XFeat(const XFeat&) = delete;
    XFeat& operator=(const XFeat&) = delete;
};