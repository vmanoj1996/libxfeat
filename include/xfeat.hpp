// Copyright 2025 Manoj Velmurugan
// SPDX-License-Identifier: MIT

#pragma once
#include "xfeat_params.hpp"

#include "conv2d_gemm.hpp"
#include "fold.hpp"

#include <memory>

using XFeatOut = std::tuple<DevicePointer<FLOAT>&, DevicePointer<FLOAT>&, DevicePointer<FLOAT>&, DevicePointer<FLOAT>&>;

class XFeat
{

    public:
    XFeat(std::string model_file, int height_, int width_);
    ~XFeat();
    
    void init(DevicePointer<FLOAT>& input);
    XFeatOut forward(DevicePointer<FLOAT>& input);

    // Disable copy operations
    XFeat(const XFeat&) = delete;
    XFeat& operator=(const XFeat&) = delete;

private:
    XFeatParams model;
    const int height, width;

    // layers
    std::vector<std::unique_ptr<Layer>> kp_layers, backbone_layers, block_fusion_layers, heatmap_layers, unfold_layer;
    std::unique_ptr<Layer> skip_pool, skip_conv;
    std::unique_ptr<Layer> interp_x4_to_x3, interp_x5_to_x3;
    std::unique_ptr<Layer> add_skip, add_layer_pyramid;
    std::unique_ptr<Layer> norm_layer;

    // cuda graph
    cudaGraph_t graph;
    cudaGraphExec_t graphExec;
    bool graph_created = false;
    cudaStream_t stream;

    // block setups
    void setup_kp();
    void setup_descriptor();
    void setup_heatmap();
    void setup_block_fusion();

    // conv setup helpers
    template<int IC, int OC, int K, int S, int P> void add_backbone_layer(const std::string &block, int layer_idx)
    {
        using std::to_string;
        ImgProperty layer1Prop = {IC, height, width};
        auto layername = "net." + block + "." + to_string(layer_idx) + ".layer.";
        auto weights_name = layername + "0.weight";
        add_conv_layer<IC, OC, K, S, P>(layer1Prop, backbone_layers, BNR<OC>(model, layername + "1"), weights_name);
    }

    // templates int IC, int OC, int K, int S, int P, typename Operation
    template<int IC, int OC, int K, int S, int P, typename Operation> void add_heatmap_layer(const std::string &layername, Operation operation)
    {
        using std::to_string;
        ImgProperty layer1Prop = {IC, height/8, width/8};
        auto weights_name = "net." + layername + ".weight";
        add_conv_layer<IC, OC, K, S, P>(layer1Prop, heatmap_layers, operation, weights_name);
    }
    template<int IC, int OC, int K, int S, int P, typename Operation> void add_fusion_layer(const std::string &layername, Operation operation)
    {
        using std::to_string;
        ImgProperty layer1Prop = {IC, height/8, width/8};
        auto weights_name = "net." + layername + ".weight";
        add_conv_layer<IC, OC, K, S, P>(layer1Prop, block_fusion_layers, operation, weights_name);
    }

    template<int IC, int OC, int K, int S, int P, typename operation> void add_conv_layer(ImgProperty layer1Prop, std::vector<std::unique_ptr<Layer>>& layers, operation op, std::string weights_name);

    // graph and output
    DevicePointer<FLOAT>* heatmap_ref;
    DevicePointer<FLOAT>* keypoints_folded_ref;  
    DevicePointer<FLOAT>* keypoints_ref;
    DevicePointer<FLOAT>* feats_ref;
    void create_cuda_graph(DevicePointer<FLOAT>& sample_input);
    XFeatOut forward_impl(DevicePointer<FLOAT>& input);
};

template<int IC, int OC, int K, int S, int P, typename operation>
inline void XFeat::add_conv_layer(ImgProperty layer1Prop, std::vector<std::unique_ptr<Layer>>& layers, operation op, std::string weights_name) 
{
    ImgProperty input_spec = layer1Prop;
    if (!layers.empty())
    {
        auto output_spec = layers.back()->get_output_spec();
        input_spec = {IC, output_spec.height, output_spec.width};
    }

    constexpr Conv2DParams params{K, K, IC, OC, S, S, P, P};
    layers.emplace_back(
        conv2d<params>(
            input_spec,
            model.getParam(weights_name),
            op, 
            stream));
}

