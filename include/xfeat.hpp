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

    // layers
    std::vector<std::unique_ptr<Layer>> kp_layers, backbone_layers, block_fusion_layers, heatmap_layers, unfold_layer ;
    std::unique_ptr<Layer> skip_pool, skip_conv;
    std::unique_ptr<Layer> interp_x4_to_x3, interp_x5_to_x3;
    std::unique_ptr<Layer> add_skip, add_layer_pyramid;
    std::unique_ptr<Layer> norm_layer;

    void setup_kp();
    void setup_descriptor();
    void setup_heatmap();
    void setup_block_fusion();

};