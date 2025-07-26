#include<primitives.hpp>
#include<memory>
#include <cuda_runtime.h>

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

    const DevicePointer<FLOAT>& forward(const DevicePointer<FLOAT>& input) override
    {
        // Apply activation element-wise
        apply_activation_kernel<<<grid, block>>>(input.get(), output_device.get(), input_prop.total_size(), op);
        return output_device;
    }
};

template<typename Operation>
inline std::unique_ptr<Layer> activation(ImgProperty input_prop, Operation op)
{
   return std::make_unique<ActivationLayer<Operation>>(input_prop, op);
}