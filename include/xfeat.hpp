#pragma once
#include "xfeat_params.hpp"

#include "conv2d.hpp"
#include "fold.hpp"

#include <memory>

class XFeat
{
private:
    XFeatParams model;
    const int height, width;
    
    // Keypoint layers
    Fold2D folder;
    Fold2D unfolder;

    std::vector<std::unique_ptr<Conv2D>> kp_layers;


public:
    XFeat(std::string model_file, int height_, int width_);
    ~XFeat() = default;
    
    DevicePointer<FLOAT>& forward(DevicePointer<FLOAT>& input);
    
    // Disable copy operations
    XFeat(const XFeat&) = delete;
    XFeat& operator=(const XFeat&) = delete;
};