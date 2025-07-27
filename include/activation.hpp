#include<primitives.hpp>
#include<memory>
#include <cuda_runtime.h>
#include "device_ops.hpp"

template<typename Operation>
class ActivationLayer : public Layer
{
private:
    Operation op;
    DevicePointer<FLOAT> output_device;
    ImgProperty input_prop, output_prop;

public:
    ActivationLayer(ImgProperty input_prop_, Operation op_) : input_prop(input_prop_), output_prop(input_prop_), op(op_)
    {
        std::vector<int> shape = {input_prop.channels, input_prop.height, input_prop.width};
        output_device.alloc(shape);
    }

    ~ActivationLayer() = default;

    virtual DevicePointer<FLOAT>& forward(const DevicePointer<FLOAT>& input) override;

};

template<typename Operation>
inline std::unique_ptr<Layer> activation(ImgProperty input_prop, Operation op)
{
   return std::make_unique<ActivationLayer<Operation>>(input_prop, op);
}

template class ActivationLayer<Sigmoid>;