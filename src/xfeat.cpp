#include "xfeat.hpp"
#include "conv2d.hpp"
#include "fold.hpp"
#include <opencv2/opencv.hpp>

XFeat::XFeat(std::string model_file, int height_, int width_) : model(model_file), height(height_), width(width_)
{
    using std::to_string;

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
                    BNR(model.getParam(layername + "1.running_mean"), model.getParam(layername + "1.running_var"))));
        }

        // Final conv layer
        kp_layers.emplace_back(
            conv2d(
                {KP_CH, cheight, cwidth},
                {1, 1, KP_CH, KP_CH + 1, 1, 1, 0, 0},
                model.getParam("net.keypoint_head.3.weight")));

        kp_layers.emplace_back(make_unfold(height, width));
    }
}

DevicePointer<FLOAT> &XFeat::forward(DevicePointer<FLOAT> &input)
{
    auto *previous_output = &input;

    // run through the keypoint layers
    int count = 0;
    for (auto &layer : kp_layers)
    {
        std::cout << "Inference: KP: Layer " << count++ << "\n";
        // although I am taking the const, trust Manoj lol
        auto *output = const_cast<DevicePointer<float> *>(&(layer->forward(*previous_output)));
        previous_output = output;
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