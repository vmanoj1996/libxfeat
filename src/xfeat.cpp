#include "xfeat.hpp"
#include "conv2d.hpp"
#include "fold.hpp"
#include <opencv2/opencv.hpp>


XFeat::XFeat(std::string model_file, int height_, int width_): model(model_file), height(height_), width(width_)
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

    kp_layers.emplace_back(make_fold(height, width));

    int cheight = height/8;
    int cwidth  = width/8;

    std::cout<<"Added fold layer\n";

    const int KP_CHANNEL = 64;

    for(int i=0; i<3; i++)
    {
        ImgProperty img_property(KP_CHANNEL, cheight, cwidth);
        auto weight = model.getParam("net.keypoint_head." + to_string(i) + ".layer.0.weight");
        Conv2DParams params(1, 1, KP_CHANNEL, KP_CHANNEL, 1, 1, 0, 0);

        auto mean = model.getParam("net.keypoint_head." + to_string(i) + ".layer.1.running_mean");
        auto vars = model.getParam("net.keypoint_head." + to_string(i) + ".layer.1.running_var");
        
        kp_layers.emplace_back(conv2d(img_property, params, weight, BNR(mean, vars)));

        std::cout<<"Added conv2d layer\n";

    }

    auto img_property = ImgProperty(KP_CHANNEL, cheight, cwidth);
    auto weight = model.getParam("net.keypoint_head.3.weight");
    Conv2DParams params(1, 1, KP_CHANNEL, KP_CHANNEL+1, 1, 1, 0, 0);

    kp_layers.emplace_back(conv2d(img_property, params, weight)); std::cout<<"Added conv2d layer\n";

    kp_layers.emplace_back(make_unfold(height, width));     std::cout<<"Added unfold layer\n";

}

DevicePointer<FLOAT>& XFeat::forward(DevicePointer<FLOAT>& input)
{
    auto* previous_output = &input;
    
    // run through the keypoint layers
    int count = 0;
    for(auto& layer: kp_layers)
    {
        std::cout<<"Inference: KP: Layer "<<count++<<"\n";
        // although I am taking the const, trust Manoj lol
        auto* output = const_cast<DevicePointer<float>*>( &(layer->forward(*previous_output)) );
        previous_output = output;
    }

    return *previous_output;
}


int main()
{
    cv::Mat img = cv::imread("../data/TajMahal.png", cv::IMREAD_GRAYSCALE);
    cv::Mat resized_img; cv::resize(img, resized_img, cv::Size(64, 32));

    cv::Mat img_float;
    resized_img.convertTo(img_float, CV_32F, 1.0/255.0);


    std::vector<float> img_vec(img_float.begin<float>(), img_float.end<float>());
    std::vector<int> dims = {img.rows, img.cols};

    DevicePointer<float> img_device(img_vec, dims);

    XFeat feat("../params/xfeat_weights.h5", img.rows, img.cols);

    feat.forward(img_device);
    

    std::cout<<"Reached the end of main\n";

}