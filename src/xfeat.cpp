
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

   // Block1: 1->4->8->8->24
   std::vector<int> block1_channels = {1, 4, 8, 8, 24};
   std::vector<int> block1_strides = {1, 2, 1, 2};
   int h = height, w = width;

   for (int i = 0; i < 4; i++)
   {
       auto layername = "net.block1." + to_string(i) + ".layer.";
       backbone_layers.emplace_back(conv2d(
           {block1_channels[i], h, w},
           {3, 3, block1_channels[i], block1_channels[i + 1], block1_strides[i], block1_strides[i], 1, 1},
           model.getParam(layername + "0.weight"),
           BNR(model, layername + "1")));

       if (block1_strides[i] == 2) { h /= 2; w /= 2; }
   }

   // Block2: 24->24 (2 layers)
   for (int i = 0; i < 2; i++)
   {
       auto layername = "net.block2." + to_string(i) + ".layer.";
       backbone_layers.emplace_back(conv2d(
           {24, h, w}, {3, 3, 24, 24, 1, 1, 1, 1},
           model.getParam(layername + "0.weight"),
           BNR(model, layername + "1")));
   }

   // Block3: 24->64->64->64
   std::vector<int> block3_channels = {24, 64, 64, 64};
   std::vector<int> block3_kernels = {3, 3, 1};
   std::vector<int> block3_strides = {2, 1, 1};

   for (int i = 0; i < 3; i++)
   {
       auto layername = "net.block3." + to_string(i) + ".layer.";
       backbone_layers.emplace_back(conv2d(
           {block3_channels[i], h, w},
           {block3_kernels[i], block3_kernels[i], block3_channels[i], block3_channels[i + 1],
            block3_strides[i], block3_strides[i], block3_kernels[i] / 2, block3_kernels[i] / 2},
           model.getParam(layername + "0.weight"),
           BNR(model, layername + "1")));

       if (block3_strides[i] == 2) { h /= 2; w /= 2; }
   }

   // Block4: 64->64->64->64 (3 layers)
   for (int i = 0; i < 3; i++)
   {
       auto layername = "net.block4." + to_string(i) + ".layer.";
       int stride = (i == 0) ? 2 : 1;
       backbone_layers.emplace_back(conv2d(
           {64, h, w}, {3, 3, 64, 64, stride, stride, 1, 1},
           model.getParam(layername + "0.weight"),
           BNR(model, layername + "1")));

       if (stride == 2) { h /= 2; w /= 2; }
   }

   // Block5: 64->128->128->128->64
   std::vector<int> block5_channels = {64, 128, 128, 128, 64};
   std::vector<int> block5_kernels = {3, 3, 3, 1};
   std::vector<int> block5_strides = {2, 1, 1, 1};

   for (int i = 0; i < 4; i++)
   {
       auto layername = "net.block5." + to_string(i) + ".layer.";
       backbone_layers.emplace_back(conv2d(
           {block5_channels[i], h, w},
           {block5_kernels[i], block5_kernels[i], block5_channels[i], block5_channels[i + 1],
            block5_strides[i], block5_strides[i], block5_kernels[i] / 2, block5_kernels[i] / 2},
           model.getParam(layername + "0.weight"),
           BNR(model, layername + "1")));

       if (block5_strides[i] == 2) { h /= 2; w /= 2; }
   }

   // TODO: AvgPool2d for skip1
   // TODO: F.interpolate + element-wise add for pyramid fusion
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

DevicePointer<FLOAT> &XFeat::forward(DevicePointer<FLOAT> &input)
{
    // Normalize the input
    DevicePointer<FLOAT> norm_output(input);
    image_norm_2d(input.get(), norm_output.get(), height, width, 1e-5f);

    auto *current_output = &norm_output;
    
    // Storage for intermediate outputs
    DevicePointer<FLOAT> *x3_output = nullptr;
    DevicePointer<FLOAT> *x4_output = nullptr;
    
    // Run through backbone layers with intermediate storage
    int backbone_idx = 0;
    
    // Block1 (4 layers) + Block2 (2 layers) = 6 layers
    for (int i = 0; i < 6; i++) {
        auto *output = const_cast<DevicePointer<float> *>(&(backbone_layers[backbone_idx]->forward(*current_output)));
        current_output = output;
        backbone_idx++;
    }
    
    // Block3 (3 layers) - store output after first layer (stride=2)
    auto *output = const_cast<DevicePointer<float> *>(&(backbone_layers[backbone_idx]->forward(*current_output)));
    current_output = output;
    backbone_idx++;
    
    // Continue block3
    for (int i = 1; i < 3; i++) {
        auto *output = const_cast<DevicePointer<float> *>(&(backbone_layers[backbone_idx]->forward(*current_output)));
        current_output = output;
        backbone_idx++;
    }
    x3_output = current_output; // Store x3 output
    
    // Block4 (3 layers)
    for (int i = 0; i < 3; i++) {
        auto *output = const_cast<DevicePointer<float> *>(&(backbone_layers[backbone_idx]->forward(*current_output)));
        current_output = output;
        backbone_idx++;
    }
    x4_output = current_output; // Store x4 output
    
    // Block5 (4 layers)
    for (int i = 0; i < 4; i++) {
        auto *output = const_cast<DevicePointer<float> *>(&(backbone_layers[backbone_idx]->forward(*current_output)));
        current_output = output;
        backbone_idx++;
    }
    auto *x5_output = current_output; // x5 output
    
    // Interpolate x4 and x5 to x3 size
    auto *x4_interp = const_cast<DevicePointer<float> *>(&(interp_x4_to_x3->forward(*x4_output)));
    auto *x5_interp = const_cast<DevicePointer<float> *>(&(interp_x5_to_x3->forward(*x5_output)));
    
    // Element-wise addition: x3 + x4_interp + x5_interp
    
    std::vector<const DevicePointer<FLOAT>*> pyramid_inputs = {x3_output, x4_interp, x5_interp};
    auto *pyramid_sum = const_cast<DevicePointer<float>*>(&(add_layer_pyramid->forward(pyramid_inputs)));
    auto *fusion_input = pyramid_sum;   

    // Block fusion
    current_output = fusion_input;
    for (auto &layer : block_fusion_layers)
    {
        auto *output = const_cast<DevicePointer<float> *>(&(layer->forward(*current_output)));
        current_output = output;
    }
    auto *feats = current_output;
    
    // Heatmap head
    current_output = feats;
    for (auto &layer : heatmap_layers)
    {
        auto *output = const_cast<DevicePointer<float> *>(&(layer->forward(*current_output)));
        current_output = output;
    }
    auto *heatmap = current_output;
    
    // Keypoint head (using original normalized input with fold/unfold)
    current_output = &norm_output;
    for (auto &layer : kp_layers)
    {
        auto *output = const_cast<DevicePointer<float> *>(&(layer->forward(*current_output)));
        current_output = output;
    }
    auto *keypoints = current_output;
    
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