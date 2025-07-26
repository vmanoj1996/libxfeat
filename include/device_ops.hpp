#pragma once

#include <cuda_runtime.h>
#include <cmath>


// struct DeviceOp
// {
// public:
//     virtual __device__ float forward(float u) { return u; }
//     virtual __device__ float forward(float u, int index) { return u; }

//     virtual __device__ void deleter() {}
// };

struct BatchNormRelu
{
    // all the pointers should be on the device
public:
    float *mean;
    float *var;
    int N;
    const float eps = 1e-5;

    __device__ float forward(float u, int buffer_index)
    {
        float y = 0.0f;
        if (buffer_index < N)
        {
            y = (u - mean[buffer_index]) / (sqrtf(var[buffer_index]) + eps);
            y = (y > 0) ? y : 0.0f;
        }

        return y;
    }

    // Factory functions
    static inline BatchNormRelu create(const std::vector<float>& host_mean, const std::vector<float>& host_var) 
    {
        BatchNormRelu op;
        op.N = host_mean.size();
        
        cudaMalloc(&op.mean, op.N * sizeof(float));
        cudaMalloc(&op.var, op.N * sizeof(float));
        
        cudaMemcpy(op.mean, host_mean.data(), op.N * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(op.var, host_var.data(), op.N * sizeof(float), cudaMemcpyHostToDevice);
        
        return op;
    }
    
    inline void destroy() 
    {
        if (mean) { cudaFree(mean); mean = nullptr; }
        if (var) { cudaFree(var); var = nullptr; }
    }

};

inline BatchNormRelu BNR(const std::vector<float>& mean, const std::vector<float>& var) 
{
    // shortform for batch norm relu
    return BatchNormRelu::create(mean, var);
}

template<typename Model>
inline BatchNormRelu BNR(Model model, std::string layername) 
{
    return BatchNormRelu::create(model.getParam(layername + ".running_mean"), model.getParam(layername + ".running_var"));
}

struct BiasOp
{
    // all the pointers should be on the device
public:
    float *bias;
    int N;
    const float eps = 1e-5;

    __device__ float forward(float u, int buffer_index)
    {
        float y = 0.0f;
        if (buffer_index < N)
        {
            y = u + bias[buffer_index];
        }

        return y;
    }

    // Factory functions
    static inline BiasOp create(const std::vector<float>& host_data) 
    {
        BiasOp op;
        op.N = host_data.size();
        
        cudaMalloc(&op.bias, op.N * sizeof(float));
        cudaMemcpy(op.bias, host_data.data(), op.N * sizeof(float), cudaMemcpyHostToDevice);
        
        return op;
    }
    
    inline void destroy() 
    {
        if (bias) { cudaFree(bias); bias = nullptr; }
    }
};

inline BiasOp Bias(const std::vector<float>& data) 
{
    return BiasOp::create(data);
}

struct Sigmoid
{
public:
    __device__ float forward(float u, int buffer_index)
    {
        return 1.0f / (1.0f + expf(-u));
    }

    inline void destroy() {}
};


struct ScaleRelu
{
public:
    float bias = 0.0f;
    float scale = 1.0f;

    __device__ float forward(float u, int idx = 0)
    {
        float y = scale * u + bias;
        return (y > 0) ? y : 0.0f;
    }
    __device__ void deleter(){}
};

struct Identity
{
public:
    __device__ float forward(float u, int idx = 0)
    {
        return u;
    }

    inline void destroy() 
    {

    }
};