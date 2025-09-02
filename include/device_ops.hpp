// Copyright 2025 Manoj Velmurugan
// SPDX-License-Identifier: MIT

#pragma once

#include <cuda_runtime.h>
#include <cmath>

template<int N>
struct BatchNormReluTemplated
{
    float mean[N];
    float rsqrt_var[N];
    static constexpr float eps = 1e-5f;

    __device__ __forceinline__ float forward(float u, int buffer_index) const
    {
        float y = (u - mean[buffer_index]) * rsqrt_var[buffer_index];
        return fmaxf(y, 0.0f);
    }

    static inline BatchNormReluTemplated<N> create(
        const std::vector<float>& host_mean, 
        const std::vector<float>& host_var) 
    {
        BatchNormReluTemplated<N> op;
        
        for(int i = 0; i < N; i++) {
            op.mean[i] = host_mean[i];
            op.rsqrt_var[i] = 1.0f / sqrtf(host_var[i] + eps);
        }
        
        return op;
    }

    inline void destroy() {} // No GPU memory to free
};

/*
struct BatchNormRelu
{
    // all the pointers should be on the device
public:
    float *mean;
    float *var; //unused TODO remove it
    float *rsqrt_var;
    int N;
    const float eps = 1e-5;

    __device__ inline float forward(float u, int buffer_index) const;

    // Factory functions
    static inline BatchNormRelu create(const std::vector<float>& host_mean, const std::vector<float>& host_var) 
    {
        BatchNormRelu op;
        op.N = host_mean.size();
        
        cudaMalloc(&op.mean, op.N * sizeof(float));
        cudaMalloc(&op.var,  op.N * sizeof(float));
        cudaMalloc(&op.rsqrt_var, op.N * sizeof(float));
        
        cudaMemcpy(op.mean, host_mean.data(), op.N * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(op.var, host_var.data(), op.N * sizeof(float), cudaMemcpyHostToDevice);

        std::vector<float> host_rsqvar(host_var.size());
        for(int i=0; i<host_var.size(); i++)
        {
            host_rsqvar[i] = 1.0f / sqrtf(host_var[i] + op.eps);
        }
        cudaMemcpy(op.rsqrt_var, host_rsqvar.data(), op.N * sizeof(float), cudaMemcpyHostToDevice);
        
        return op;
    }
    
    inline void destroy() 
    {
        if (mean) { cudaFree(mean); mean = nullptr; }
        if (var) { cudaFree(var); var = nullptr; }
        if (rsqrt_var) { cudaFree(rsqrt_var); rsqrt_var = nullptr; }
    }

};
*/
struct BiasOp
{
    // all the pointers should be on the device
public:
    float *bias;
    int N;
    const float eps = 1e-5;

    __device__ inline float forward(float u, int buffer_index) const
    {
        return (buffer_index < N)? u + bias[buffer_index]:0.0f;;
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


struct Sigmoid
{
public:
    __device__ inline float forward(float u, int buffer_index) const
    {
        return 1.0f / (1.0f + __expf(-u)); // may have lower precision? TODO check this
    }

    inline void destroy() {}
};


struct ScaleRelu
{
public:
    float bias = 0.0f;
    float scale = 1.0f;

    __device__ float forward(float u, int idx = 0) const
    {
        float y = scale * u + bias;
        return (y > 0) ? y : 0.0f;
    }
    __device__ void destroy(){}
};

struct Identity
{
public:
    __device__ inline float forward(float u, int idx = 0) const
    {
        return u;
    }

    inline void destroy() 
    {

    }
};

/*
#ifdef __CUDACC__
__device__ inline float BatchNormRelu::forward(float u, int buffer_index) const
{
    float y = 0.0f;
    // y = (buffer_index < N)? (u - mean[buffer_index]) * rsqrtf(var[buffer_index] + eps):0.0f;
    y = (u - mean[buffer_index]) * rsqrt_var[buffer_index];
    y = fmaxf(y, 0.0f);

    return y;
}
#endif

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
*/

template<int N>
inline BatchNormReluTemplated<N> BNR(const std::vector<float>& mean, const std::vector<float>& var) 
{
    return BatchNormReluTemplated<N>::create(mean, var);
}

template<int N, typename Model>
inline BatchNormReluTemplated<N> BNR(Model model, std::string layername) 
{
    return BatchNormReluTemplated<N>::create(
        model.getParam(layername + ".running_mean"), 
        model.getParam(layername + ".running_var")
    );
}

inline BiasOp Bias(const std::vector<float>& data) 
{
    return BiasOp::create(data);
}
