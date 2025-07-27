
// https://github.com/verlab/accelerated_features/blob/main/modules/model.py
#include "xfeat.hpp"
#include "conv2d.hpp"
#include "fold.hpp"
#include <opencv2/opencv.hpp>
#include <fstream>
#include "primitives.hpp"
#include "add.hpp" 
#include "activation.hpp" 
#include "pool.hpp" 
#include "interp.hpp" 

void XFeat::setup_kp()
{
    using std::to_string;

    if (kp_layers.size() != 0)
    {
        std::cout << "already kp_layers populated\n";
        return; // already done
    }

    kp_layers.emplace_back(make_fold(height, width));

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
                BNR(model, layername + "1")));
    }

    // Final conv layer
    kp_layers.emplace_back(
        conv2d(
            {KP_CH, cheight, cwidth},
            {1, 1, KP_CH, KP_CH + 1, 1, 1, 0, 0},
            model.getParam("net.keypoint_head.3.weight"),
            Bias(model.getParam("net.keypoint_head.3.bias"))));

    kp_layers.emplace_back(make_unfold(height, width));
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
   auto add_conv_layer = [&](const std::string& block, int layer_idx, int in_ch, int out_ch, int k, int stride, int padding) 
   {
       auto layername = "net." + block + "." + to_string(layer_idx) + ".layer.";
       
       // Get current dimensions
       int h, w;
       if (backbone_layers.empty()) {
           h = height;
           w = width;
       } else {
           // Calculate dimensions based on previous layer
           // For now, we'll track manually - you can implement get_output_spec() later
           static int current_h = height, current_w = width;
           h = current_h;
           w = current_w;
           
           // Update for next iteration
           if (stride == 2) {
               current_h /= 2;
               current_w /= 2;
           }
       }

       backbone_layers.emplace_back(conv2d(
           {in_ch, h, w}, {k, k, in_ch, out_ch, stride, stride, padding, padding},
           model.getParam(layername + "0.weight"),
           BNR(model, layername + "1")));
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
   add_conv_layer("block5", 0, 64, 128, 3, 2, 1);
   add_conv_layer("block5", 1, 128, 128, 3, 1, 1);
   add_conv_layer("block5", 2, 128, 128, 3, 1, 1);
   add_conv_layer("block5", 3, 128, 64, 1, 1, 0);
}

void XFeat::setup_heatmap()
{
   using std::to_string;
   
   if (heatmap_layers.size() != 0)
   {
       std::cout << "already heatmap_layers populated\n";
       return;
   }

   auto [cheight, cwidth] = std::make_pair(height / 8, width / 8);
   const int HM_CH = 64;

   // 2 BasicLayers with 1x1 conv + BatchNorm + ReLU
   for (int i = 0; i < 2; i++)
   {
       auto layername = "net.heatmap_head." + to_string(i) + ".layer.";
       heatmap_layers.emplace_back(conv2d(
           {HM_CH, cheight, cwidth},
           {1, 1, HM_CH, HM_CH, 1, 1, 0, 0},
           model.getParam(layername + "0.weight"),
           BNR(model, layername + "1")));
   }

   // Final conv: 64 -> 1 channel (with bias)
   heatmap_layers.emplace_back(conv2d(
       {HM_CH, cheight, cwidth},
       {1, 1, HM_CH, 1, 1, 1, 0, 0},
       model.getParam("net.heatmap_head.2.weight"),
       Bias(model.getParam("net.heatmap_head.2.bias"))));

   // Sigmoid activation
   heatmap_layers.emplace_back(activation({1, cheight, cwidth}, Sigmoid{}));
}

void XFeat::setup_block_fusion()
{
   using std::to_string;
   
   if (block_fusion_layers.size() != 0)
   {
       std::cout << "already block_fusion_layers populated\n";
       return;
   }

   auto [cheight, cwidth] = std::make_pair(height / 8, width / 8);
   const int BF_CH = 64;

   // 2 BasicLayers with 3x3 conv + BatchNorm + ReLU, stride=1
   for (int i = 0; i < 2; i++)
   {
       auto layername = "net.block_fusion." + to_string(i) + ".layer.";
       block_fusion_layers.emplace_back(conv2d(
           {BF_CH, cheight, cwidth},
           {3, 3, BF_CH, BF_CH, 1, 1, 1, 1},
           model.getParam(layername + "0.weight"),
           BNR(model, layername + "1")));
   }

   // Final conv: 64 -> 64, 1x1 kernel (no BatchNorm/ReLU)
   block_fusion_layers.emplace_back(conv2d(
       {BF_CH, cheight, cwidth},
       {1, 1, BF_CH, BF_CH, 1, 1, 0, 0},
       model.getParam("net.block_fusion.2.weight"),
       Bias(model.getParam("net.block_fusion.2.bias"))));
}


XFeat::XFeat(std::string model_file, int height_, int width_) : model(model_file), height(height_), width(width_)
{
    setup_descriptor();
    setup_kp();
    setup_heatmap();
    setup_block_fusion();
    setup_interpolation();
}

void save_layer_data(const DevicePointer<float> &data, const std::string &name)
{
    auto host_data = data.get_value();
    auto shape = data.get_shape();

    // Save binary data
    std::ofstream file(name + "_output.bin", std::ios::binary);
    file.write(reinterpret_cast<const char *>(host_data.data()),
               host_data.size() * sizeof(float));
    file.close();

    // Save shape
    std::ofstream shape_file(name + "_shape.txt");
    for (int dim : shape)
    {
        shape_file << dim << " ";
    }
    shape_file.close();
}

void XFeat::setup_interpolation() {
    auto [x3_h, x3_w] = std::make_pair(height / 4, width / 4);  // After block3
    
    interp_x4_to_x3 = interp2d({64, height/8, width/8}, x3_h, x3_w);
    interp_x5_to_x3 = interp2d({64, height/16, width/16}, x3_h, x3_w);
    add_layer_pyramid = add_layer({64, x3_h, x3_w});
}

DevicePointer<FLOAT> &XFeat::forward(DevicePointer<FLOAT> &input) {
    // Normalize the input
    DevicePointer<FLOAT> norm_output(input);
    image_norm_2d(input.get(), norm_output.get(), height, width, 1e-5f);
    
    auto *curr_out_ptr = &norm_output;
    
    // Storage for intermediate outputs
    DevicePointer<FLOAT> *x3_output = nullptr;
    DevicePointer<FLOAT> *x4_output = nullptr;
    
    // Run through backbone layers with intermediate storage
    int backbone_idx = 0;
    
    // Block1 (4 layers) + Block2 (2 layers) = 6 layers
    for (int i = 0; i < 6; i++) {
        auto output = backbone_layers[backbone_idx]->forward(*curr_out_ptr);
        curr_out_ptr = &output;
        backbone_idx++;
    }
    
    // Block3 (3 layers) - store output after first layer (stride=2)
    auto output = backbone_layers[backbone_idx]->forward(*curr_out_ptr);
    curr_out_ptr = &output;
    backbone_idx++;
    
    // Continue block3
    for (int i = 1; i < 3; i++) {
        auto output = backbone_layers[backbone_idx]->forward(*curr_out_ptr);
        curr_out_ptr = &output;
        backbone_idx++;
    }
    x3_output = curr_out_ptr; // Store x3 output
    
    // Block4 (3 layers)
    for (int i = 0; i < 3; i++) {
        auto output = backbone_layers[backbone_idx]->forward(*curr_out_ptr);
        curr_out_ptr = &output;
        backbone_idx++;
    }
    x4_output = curr_out_ptr; // Store x4 output
    
    // Block5 (4 layers)
    for (int i = 0; i < 4; i++) {
        auto output = backbone_layers[backbone_idx]->forward(*curr_out_ptr);
        curr_out_ptr = &output;
        backbone_idx++;
    }
    auto *x5_output = curr_out_ptr; // x5 output
    
    // Interpolate x4 and x5 to x3 size
    auto x4_interp = interp_x4_to_x3->forward(*x4_output);
    auto x5_interp = interp_x5_to_x3->forward(*x5_output);
    
    // Element-wise addition: x3 + x4_interp + x5_interp
    std::vector<const DevicePointer<FLOAT>*> pyramid_inputs = {x3_output, &x4_interp, &x5_interp};
    auto pyramid_sum = add_layer_pyramid->forward(pyramid_inputs);
    curr_out_ptr = &pyramid_sum;

    // Block fusion
    for (auto &layer : block_fusion_layers) {
        auto output = layer->forward(*curr_out_ptr);
        curr_out_ptr = &output;
    }
    auto *feats = curr_out_ptr;
    
    // Heatmap head
    curr_out_ptr = feats;
    for (auto &layer : heatmap_layers) {
        auto output = layer->forward(*curr_out_ptr);
        curr_out_ptr = &output;
    }
    auto *heatmap = curr_out_ptr;
    
    // Keypoint head (using original normalized input with fold/unfold)
    curr_out_ptr = &norm_output;
    for (auto &layer : kp_layers) {
        auto output = layer->forward(*curr_out_ptr);
        curr_out_ptr = &output;
    }
    auto *keypoints = curr_out_ptr;
    
    return *feats;
}

int main()
{
    cv::Mat img = cv::imread("../data/TajMahal.png", cv::IMREAD_GRAYSCALE);
    // cv::Mat resized_img; cv::resize(img, resized_img, cv::Size(64, 32));

    cv::Mat img_float;
    img.convertTo(img_float, CV_32F, 1.0 / 255.0);

    std::vector<float> img_vec(img_float.begin<float>(), img_float.end<float>());
    std::vector<int> dims = {img.rows, img.cols};

    DevicePointer<float> img_device(img_vec, dims);

    XFeat feat("../params/xfeat_weights.h5", img.rows, img.cols);

    feat.forward(img_device);

    std::cout << "Reached the end of main\n";
}