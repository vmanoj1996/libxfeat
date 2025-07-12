
#include "xfeat.hpp"

matx::tensor_t<float, 4> XFeat::keypointHead()
{
    using matx::make_tensor;

    // Allocate the storage on GPU for input and output
    auto inputs = make_tensor<float>({10, 3, 480, 640});
    // auto inputs_folded;

    // auto kp_kernel = make_tensor<float>({3, 1, 1});  // 1 output channel

    // auto results = make_tensor<float>({10, 1, 480, 640});
    // auto results_unfolded = make_tensor<float>({10, 1, 480, 640});

    // int layer = 0;
    // for(layer=0; layer<3; layer++)
    // {
    //     // conv operation

    //     // batch norm

    //     // relu operation
    // }

    // //
    // layer = 3;
    // // output convolution

    // // do the unfold operation

    // // return the result

    // (result = conv2d(images, kp_kernel, matx::MATX_C_MODE_FULL)).run(executor_);

    return inputs;
}