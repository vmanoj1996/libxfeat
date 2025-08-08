// Copyright 2025 Manoj Velmurugan
// SPDX-License-Identifier: MIT

// C++ and CUDA only XFEAT Feature Extractor Implementation

// https://github.com/verlab/accelerated_features/blob/main/modules/model.py

#include "xfeat.hpp"
#include "conv2d.hpp"
#include "fold.hpp"
#include <fstream>
#include "primitives.hpp"
#include "add.hpp"
#include "activation.hpp"
#include "pool.hpp"
#include "interp.hpp"
#include "normalize.hpp"
#include <tensorio.hpp>


inline void save_layer_data(const DevicePointer<float> &data, const std::string &name)
{
#ifdef ENABLE_XFEAT_DEBUG
    auto host_data = data.get_value();
    auto shape = data.get_shape();

    // Create debug directory if it doesn't exist
    tio::mkdir("./debug_outputs");
    
    // Convert shape to vector<int> for H5 saving
    std::vector<int> h5_shape(shape.begin(), shape.end());
    
    // Save to H5 format in debug subfolder
    std::string filename = "./debug_outputs/" + name + ".h5";
    tio::save_hdf5(host_data, h5_shape, filename, "data");
#endif

}

XFeat::XFeat(std::string model_file, int height_, int width_) : model(model_file), height(height_), width(width_)
{
    // create a execution lane on gpu
    cudaStreamCreate(&stream);

    norm_layer = image_norm_2d({1, height, width}, 1e-5f, stream);
    setup_descriptor();
    setup_kp();
    setup_heatmap();
    setup_block_fusion();
}

void XFeat::create_cuda_graph(DevicePointer<FLOAT>& sample_input) 
{
    if (graph_created) return;

    cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);
    auto [h, kf, k, f] = forward_impl(sample_input);
    heatmap_ref = &h;
    keypoints_folded_ref = &kf;
    keypoints_ref = &k;
    feats_ref = &f;
    cudaStreamEndCapture(stream, &graph); // capture the graph

    // create an executable graph instance
    cudaGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0);
    
    graph_created = true;
}

XFeatOut XFeat::forward(DevicePointer<FLOAT>& input) 
{
    if (!graph_created) 
    {
        create_cuda_graph(input);
    }
    
    cudaGraphLaunch(graphExec, stream);
    cudaStreamSynchronize(stream);
    
    return std::tie(*heatmap_ref, *keypoints_folded_ref, *keypoints_ref, *feats_ref);
}

void XFeat::setup_kp()
{
    using std::to_string;

    if (kp_layers.size() != 0)
    {
        std::cout << "already kp_layers populated\n";
        return; // already done
    }

    kp_layers.emplace_back(make_fold(height, width, 8, stream));

    auto [cheight, cwidth] = std::make_pair(height / 8, width / 8);
    const int KP_CH = 64;

    // 3 conv layers with BatchNorm+ReLU
    for (int i = 0; i < 3; i++)
    {
        auto layername = "net.keypoint_head." + to_string(i) + ".layer.";
        kp_layers.emplace_back(
            conv2d(
                {KP_CH, cheight, cwidth},
                {1, 1, KP_CH, KP_CH, 1, 1, 0, 0},
                model.getParam(layername + "0.weight"),
                BNR(model, layername + "1"),
                stream));
    }

    // Final conv layer
    kp_layers.emplace_back(
        conv2d(
            {KP_CH, cheight, cwidth},
            {1, 1, KP_CH, KP_CH + 1, 1, 1, 0, 0},
            model.getParam("net.keypoint_head.3.weight"),
            Bias(model.getParam("net.keypoint_head.3.bias")), 
            stream
        ));

    unfold_layer.emplace_back(make_unfold(height, width, 8, stream));
}
void XFeat::setup_descriptor()
{
    using std::to_string;

    if (backbone_layers.size() != 0)
    {
        std::cout << "already backbone_layers populated\n";
        return;
    }

    //  helper to make layers with less pain
    auto add_conv_layer = [&](const std::string &block, int layer_idx, int in_ch, int out_ch, int k, int stride, int padding)
    {
        auto layername = "net." + block + "." + to_string(layer_idx) + ".layer.";

        // Get current dimensions
        int h, w;
        if (backbone_layers.empty())
        {
            h = height;
            w = width;
        }
        else
        {
            // Calculate dimensions based on previous layer
            // For now, we'll track manually - you can implement get_output_spec() later
            auto output_spec = backbone_layers.back()->get_output_spec();
            h = output_spec.height;
            w = output_spec.width;
        }

        backbone_layers.emplace_back(conv2d(
            {in_ch, h, w}, {k, k, in_ch, out_ch, stride, stride, padding, padding},
            model.getParam(layername + "0.weight"),
            BNR(model, layername + "1"), 
            stream));
    };

    // Block1: 1->4->8->8->24
    add_conv_layer("block1", 0, 1, 4, 3, 1, 1);
    add_conv_layer("block1", 1, 4, 8, 3, 2, 1);
    add_conv_layer("block1", 2, 8, 8, 3, 1, 1);
    add_conv_layer("block1", 3, 8, 24, 3, 2, 1);

    // Block2: 24->24 (2 layers)
    add_conv_layer("block2", 0, 24, 24, 3, 1, 1);
    add_conv_layer("block2", 1, 24, 24, 3, 1, 1);

    // Block3: 24->64->64->64
    add_conv_layer("block3", 0, 24, 64, 3, 2, 1);
    add_conv_layer("block3", 1, 64, 64, 3, 1, 1);
    add_conv_layer("block3", 2, 64, 64, 1, 1, 0);

    // Block4: 64->64->64->64 (3 layers)
    add_conv_layer("block4", 0, 64, 64, 3, 2, 1);
    add_conv_layer("block4", 1, 64, 64, 3, 1, 1);
    add_conv_layer("block4", 2, 64, 64, 3, 1, 1);

    // Block5: 64->128->128->128->64
    add_conv_layer("block5", 0, 64,  128, 3, 2, 1);
    add_conv_layer("block5", 1, 128, 128, 3, 1, 1);
    add_conv_layer("block5", 2, 128, 128, 3, 1, 1);
    add_conv_layer("block5", 3, 128, 64,  1, 1, 0);

    skip_pool = avgpool2d({1, height, width}, PoolParams(4, 4, 4, 4, 0, 0), stream);
    skip_conv = conv2d({1, height/4, width/4}, {1, 1, 1, 24, 1, 1, 0, 0}, model.getParam("net.skip1.1.weight"), Bias(model.getParam("net.skip1.1.bias")), stream);

    interp_x4_to_x3 = interp2d(backbone_layers[12-1]->get_output_spec(), height/8, width/8, stream);
    interp_x5_to_x3 = interp2d(backbone_layers[16-1]->get_output_spec(), height/8, width/8, stream);

    add_skip = add_layer({24, height/4, width/4}, stream);
    add_layer_pyramid = add_layer(backbone_layers[9-1]->get_output_spec(), stream);
}

void XFeat::setup_heatmap()
{
    using std::to_string;

    if (heatmap_layers.size() != 0)
    {
        std::cout << "already heatmap_layers populated\n";
        return;
    }

    auto add_heatmap_layer = [&](const std::string &layer_name, int in_ch, int out_ch, int k, int stride, int padding, auto operation)
    {
        ImgProperty input_spec = {in_ch, height / 8, width / 8};
        if (!heatmap_layers.empty())
        {
            auto output_spec = heatmap_layers.back()->get_output_spec();
            input_spec = {in_ch, output_spec.height, output_spec.width};
        }

        heatmap_layers.emplace_back(conv2d(
            input_spec, {k, k, in_ch, out_ch, stride, stride, padding, padding},
            model.getParam(layer_name + ".weight"),
            operation,
            stream));
    };
    // 2 BasicLayers with 1x1 conv + BatchNorm + ReLU
    add_heatmap_layer("net.heatmap_head.0.layer.0", 64, 64, 1, 1, 0, BNR(model, "net.heatmap_head.0.layer.1"));
    add_heatmap_layer("net.heatmap_head.1.layer.0", 64, 64, 1, 1, 0, BNR(model, "net.heatmap_head.1.layer.1"));

    // Final conv: 64 -> 1 channel (with bias)
    add_heatmap_layer("net.heatmap_head.2", 64, 1, 1, 1, 0, Bias(model.getParam("net.heatmap_head.2.bias")));

    // Sigmoid activation
    auto output_spec = heatmap_layers.back()->get_output_spec();
    heatmap_layers.emplace_back(activation(output_spec, Sigmoid{}, stream));
}

void XFeat::setup_block_fusion()
{
    using std::to_string;

    if (block_fusion_layers.size() != 0)
    {
        std::cout << "already block_fusion_layers populated\n";
        return;
    }

    auto add_fusion_layer = [&](const std::string &layer_name, int in_ch, int out_ch,
                                int k, int stride, int padding, auto operation)
    {
        // Get current dimensions
        ImgProperty input_spec = {in_ch, height / 8, width / 8};
        if (!block_fusion_layers.empty())
        {
            auto output_spec = block_fusion_layers.back()->get_output_spec();
            input_spec = {in_ch, output_spec.height, output_spec.width};
        }
        block_fusion_layers.emplace_back(conv2d(
                input_spec, {k, k, in_ch, out_ch, stride, stride, padding, padding},
                model.getParam(layer_name + ".weight"),
                operation,
                stream));
    };

    // 2 BasicLayers with 3x3 conv + BatchNorm + ReLU, stride=1
    add_fusion_layer("net.block_fusion.0.layer.0", 64, 64, 3, 1, 1, BNR(model, "net.block_fusion.0.layer.1"));
    add_fusion_layer("net.block_fusion.1.layer.0", 64, 64, 3, 1, 1, BNR(model, "net.block_fusion.1.layer.1"));

    // Final conv: 64 -> 64, 1x1 kernel (no BatchNorm/ReLU)
    add_fusion_layer("net.block_fusion.2", 64, 64, 1, 1, 0, Bias(model.getParam("net.block_fusion.2.bias")));
}

XFeatOut XFeat::forward_impl(DevicePointer<FLOAT> &input)
{
    auto run_backbone = [&](int start, int count, auto& input_ref) -> auto& 
    {
        auto* current = &input_ref;
        for (int i = start; i < start + count; i++) 
        {
            current = &backbone_layers[i]->forward(*current);
            // Save backbone layer outputs
            save_layer_data(*current, "backbone_layer_" + std::to_string(i));
        }
        return *current;
    };
    
    auto run_layers = [](auto& layers, auto& input_ref) -> auto& 
    {
        auto* current = &input_ref;
        for (size_t i = 0; i < layers.size(); i++) {
            current = &layers[i]->forward(*current);
            // Save layer outputs with meaningful names
            save_layer_data(*current, "layer_" + std::to_string(i));
        }
        return *current;
    };

    // Run heads with specific naming
    auto run_named_layers = [&](auto& layers, auto& input_ref, const std::string& prefix) -> auto& 
    {
        auto* current = &input_ref;
        for (size_t i = 0; i < layers.size(); i++) {
            current = &layers[i]->forward(*current);
            save_layer_data(*current, prefix + "_layer_" + std::to_string(i));
        }
        return *current;
    };

    // Normalize the input
    auto& norm_output = norm_layer->forward(input);

    // Run backbone in chunks
    auto& x1_out = run_backbone(0, 4, norm_output);

    // Compute skip connection
    auto& skip_pooled = skip_pool->forward(norm_output);
    auto& skip_out    = skip_conv->forward(skip_pooled);

    // Add skip to x1
    std::vector<const DevicePointer<FLOAT>*> skip_inputs = {&x1_out, &skip_out};
    auto& x1_plus_skip = add_skip->forward(skip_inputs);

    // Run Block2 (next 2 layers)  
    auto& x2_out = run_backbone(4, 2, x1_plus_skip);

    // Continue with remaining blocks
    auto& x3_out = run_backbone(6, 3, x2_out);
    auto& x4_out = run_backbone(9, 3, x3_out);
    auto& x5_out = run_backbone(12, 4, x4_out);
    
    auto& x4_interp = interp_x4_to_x3->forward(x4_out);
    auto& x5_interp = interp_x5_to_x3->forward(x5_out);
    
    // Pyramid fusion: x3 + x4_interp + x5_interp
    std::vector<const DevicePointer<FLOAT>*> pyramid_inputs = {&x3_out, &x4_interp, &x5_interp};
    auto& pyramid_sum = add_layer_pyramid->forward(pyramid_inputs);

    auto& feats     = run_named_layers(block_fusion_layers, pyramid_sum, "block_fusion");
    auto& heatmap   = run_named_layers(heatmap_layers, feats, "heatmap");
    auto& keypoints_folded = run_named_layers(kp_layers, norm_output, "keypoint");
    auto& keypoints = (unfold_layer[0])->forward(keypoints_folded);

    // these are no op
    save_layer_data(norm_output, "normalized_input");
    save_layer_data(skip_pooled, "skip_pooled");
    save_layer_data(skip_out, "skip_conv_out");
    save_layer_data(x1_plus_skip, "x1_plus_skip");
    save_layer_data(x1_out, "x1_backbone_output"); 
    save_layer_data(x2_out, "x2_backbone_output");
    save_layer_data(x3_out, "x3_backbone_output");
    save_layer_data(x4_out, "x4_backbone_output");
    save_layer_data(x5_out, "x5_backbone_output");
    save_layer_data(x5_interp, "x5_interpolated");
    save_layer_data(pyramid_sum, "pyramid_fusion");
    save_layer_data(feats, "final_features");
    save_layer_data(heatmap, "final_heatmap");
    save_layer_data(keypoints, "final_keypoints");

    return std::tie(heatmap, keypoints_folded, keypoints, feats);
}
