#include "xfeat.hpp"
#include <opencv2/opencv.hpp>

int main()
{
    cv::Mat img = cv::imread("../data/TajMahal.png", cv::IMREAD_GRAYSCALE);
    // cv::Mat resized_img; cv::resize(img, resized_img, cv::Size(64, 32));

    cv::Mat img_float;
    img.convertTo(img_float, CV_32F, 1.0 / 255.0);

    std::vector<float> img_vec(img_float.begin<float>(), img_float.end<float>());
    std::vector<int> dims = {1, img.rows, img.cols};

    DevicePointer<float> img_device(img_vec, dims);

    XFeat feat("../params/xfeat_weights.h5", img.rows, img.cols);

    auto [heatmap, keypoints, feats] = feat.forward(img_device);

    std::vector<float> vec = keypoints.get_value();
    std::vector<int> shape = keypoints.get_shape();

    // cv::Mat imgout(shape[1], shape[2], CV_32F, vec.data());
    // cv::imshow("img", imgout);
    // cv::waitKey(0);

    std::cout << "Reached the end of main\n";
}