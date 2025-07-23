#pragma once

#include <cuda_runtime.h>

struct DeviceOp
{
public:
    virtual __device__ float forward(float u) { return u; }
    virtual __device__ float forward(float u, int index) { return u; }

    virtual __device__ void deleter() {}
};

struct BatchNormRelu : public DeviceOp
{
    // all the pointers should be on the device
public:
    float *mean;
    float *var;
    int N;
    float eps = 1e-5;

    __device__ float forward(float u, int buffer_index) override
    {
        float y = 0.0f;
        if (buffer_index < N)
        {
            y = (u - mean[buffer_index]) / (sqrtf(var[buffer_index] + eps));
            y = (y > 0) ? y : 0.0f;
        }

        return y;
    }

    __device__ void deleter()
    {
        if (mean)
        {
            delete[] mean;
            mean = nullptr;
        }
        if (var)
        {
            delete[] var;
            var = nullptr;
        }
    }
};

struct ScaleRelu : public DeviceOp
{
public:
    float bias = 0.0f;
    float scale = 1.0f;

    __device__ float forward(float u) override
    {
        float y = scale * u + bias;
        return (y > 0) ? y : 0.0f;
    }
};

struct Identity : public DeviceOp
{
public:
    __device__ float forward(float u) override
    {
        return u;
    }
};

__global__ void init_BatchNormRelu_kernel(DeviceOp **p);
__global__ void setup_BatchNormRelu_kernel(DeviceOp *p, float *d_mean, float *d_var, int N);
__global__ void delete_op_kernel(DeviceOp** p);

DeviceOp* create_BatchNormRelu(const std::vector<float>& mean_host, const std::vector<float>& var_host);