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

    // memory managed outside so that copy operations can work without any pain
    // BatchNormRelu(const std::vector<float>& host_mean, const std::vector<float>& host_var) : N(host_mean.size()) 
    // {
    //     cudaMalloc(&mean, N * sizeof(float));
    //     cudaMalloc(&var,  N * sizeof(float));

    //     cudaMemcpy(mean, host_mean.data(), N * sizeof(float), cudaMemcpyHostToDevice);
    //     cudaMemcpy(var,  host_var.data(), N * sizeof(float), cudaMemcpyHostToDevice);
    // }

    // ~BatchNormRelu() 
    // {
    //     // wont do anything if moved already :)
    //     if (mean) { cudaFree(mean); mean = nullptr; }
    //     if (var)  { cudaFree(var); var = nullptr; }
    // }

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

    __device__ void deleter(){}
};

__global__ void init_BatchNormRelu_kernel(BatchNormRelu **p);
__global__ void setup_BatchNormRelu_kernel(BatchNormRelu *p, float *d_mean, float *d_var, int N);
__global__ void delete_op_kernel(BatchNormRelu** p);

// factory method to create a new batch norm relu device layer
BatchNormRelu* create_BatchNormRelu(const std::vector<float>& mean_host, const std::vector<float>& var_host);