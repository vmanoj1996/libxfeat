#pragma once

#include "primitives.hpp"
#include <cuda_runtime.h>

#ifdef XFEAT_CUDA_GRAPH
    #define CUDA_SYNC_IF_NEEDED(stream) 
#else
    #define CUDA_SYNC_IF_NEEDED(stream) cudaDeviceSynchronize()
#endif

class Layer
{
    protected:
    cudaStream_t stream;

    public:
    virtual DevicePointer<FLOAT>& forward(const DevicePointer<FLOAT>& input_device) = 0;
    virtual DevicePointer<FLOAT>& forward(const std::vector<const DevicePointer<FLOAT>*>& inputs) 
    {
        static DevicePointer<FLOAT> dummy;

        throw std::runtime_error("Base class forward() should not be called");
        return dummy;
    }


    Layer() = default;
    virtual ~Layer() = default;

    virtual ImgProperty get_output_spec() const = 0;
    virtual ImgProperty get_input_spec()  const = 0;

    // Disable copy and move for all layers
    Layer(const Layer&) = delete;
    Layer& operator=(const Layer&) = delete;
    Layer(Layer&&) = delete;
    Layer& operator=(Layer&&) = delete;

};

