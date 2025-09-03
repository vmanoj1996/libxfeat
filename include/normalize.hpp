#pragma once
#include "primitives.hpp"
#include "layer.hpp"

class ImageNorm2D : public Layer 
{
private:
    DevicePointer<FLOAT> output_device;
    ImgProperty input_prop, output_prop;
    
    int size;
    
    // Pre-allocated CUB workspace
    void* d_temp_storage = nullptr;
    size_t temp_storage_bytes = 0;
    float* d_sum_result = nullptr;
    float* d_var_result = nullptr;
    
    void setup_workspace();
    void cleanup();

public:
    ImageNorm2D(ImgProperty input_prop_, float eps_, cudaStream_t stream_);
    ~ImageNorm2D();
    
    using Layer::forward;
    virtual DevicePointer<FLOAT>& forward(const DevicePointer<FLOAT>& input_device) override;
    DevicePointer<FLOAT>& get_output() { return output_device; }
    
    virtual ImgProperty get_output_spec() const override { return output_prop; }
    virtual ImgProperty get_input_spec() const override { return input_prop; }
};

inline std::unique_ptr<Layer> image_norm_2d(ImgProperty input_prop, float eps, cudaStream_t stream_) 
{
    return std::make_unique<ImageNorm2D>(input_prop, eps, stream_);
}