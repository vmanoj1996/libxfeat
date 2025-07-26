#include "xfeat.hpp"
#include "conv2d.hpp"
#include "fold.hpp"
#include <opencv2/opencv.hpp>
#include <fstream>
#include "primitives.hpp"

void XFeat::setup_kp()
{
    /* Keypoint head - separate branch using fold/unfold
        (keypoint_head): Sequential(
        (0): BasicLayer(
            (layer): Sequential(
            (0): Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=False, track_running_stats=True)
            (2): ReLU(inplace=True)
            )
        )
        (1): BasicLayer(
            (layer): Sequential(
            (0): Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=False, track_running_stats=True)
            (2): ReLU(inplace=True)
            )
        )
        (2): BasicLayer(
            (layer): Sequential(
            (0): Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=False, track_running_stats=True)
            (2): ReLU(inplace=True)
            )
        )
            (3): Conv2d(64, 65, kernel_size=(1, 1), stride=(1, 1))
    */
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

void XFeat::setup_backbone()
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
   // TODO: block_fusion layers
   // TODO: heatmap_head layers
}

XFeat::XFeat(std::string model_file, int height_, int width_) : model(model_file), height(height_), width(width_)
{
    setup_kp();
    setup_backbone();
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

DevicePointer<FLOAT> &XFeat::forward(DevicePointer<FLOAT> &input)
{
    // normalize the input
    DevicePointer<FLOAT> norm_output(input);
    image_norm_2d(input.get(), norm_output.get(), height, width, 1e-5f);

    norm_output.print_shape();
    save_layer_data(norm_output, "cpp_input");

    auto *previous_output = &norm_output;
    // run through the keypoint layers
    int count = 0;
    for (auto &layer : kp_layers)
    {
        std::cout << "Inference: KP: Layer " << count << "\n";
        // although I am taking the const, trust Manoj lol
        auto *output = const_cast<DevicePointer<float> *>(&(layer->forward(*previous_output)));
        previous_output = output;

        save_layer_data(*output, "cpp_layer_" + std::to_string(count));

        count++;
    }

    return *previous_output;
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