// Copyright 2025 Manoj Velmurugan
// SPDX-License-Identifier: MIT

#pragma once
#include "xfeat_params.hpp"

#include "conv2d.hpp"
#include "fold.hpp"

#include <memory>

class XFeat
{

    public:
    XFeat(std::string model_file, int height_, int width_);
    ~XFeat() = default;
    
    std::tuple<DevicePointer<FLOAT>&, DevicePointer<FLOAT>&, DevicePointer<FLOAT>&, DevicePointer<FLOAT>&> forward(DevicePointer<FLOAT>& input);
    
    // Disable copy operations
    XFeat(const XFeat&) = delete;
    XFeat& operator=(const XFeat&) = delete;

private:
    XFeatParams model;
    const int height, width;

    std::vector<std::unique_ptr<Layer>> kp_layers, backbone_layers, block_fusion_layers, heatmap_layers, unfold_layer ;
    std::unique_ptr<Layer> skip_pool, skip_conv;

    void setup_kp();
    void setup_descriptor();
    void setup_heatmap();
    void setup_block_fusion();

};