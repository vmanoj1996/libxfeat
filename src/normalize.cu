
// Copyright 2025 Manoj Velmurugan
// SPDX-License-Identifier: MIT

#include <thrust/reduce.h>
#include <thrust/transform_reduce.h>
#include <thrust/device_ptr.h>
#include <thrust/functional.h>

__global__ void instance_norm_kernel(const float* input, float* output, 
                                    float mean, float variance, 
                                    int size, float eps = 1e-5f)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < size) {
        float std_inv = rsqrtf(variance + eps);
        output[idx] = (input[idx] - mean) * std_inv;
    }
}

struct variance_op 
{
    float mean_val;
    variance_op(float m) : mean_val(m) {}
    __device__ float operator()(float x) const {
        float diff = x - mean_val;
        return diff * diff;
    }
};

void image_norm_2d(const float* input, float* output, int height, int width, float eps = 1e-5f)
{
    int size = height * width;
    
    // Compute mean using device_ptr
    float sum = thrust::reduce(thrust::device_ptr<const float>(input), 
                              thrust::device_ptr<const float>(input + size), 
                              0.0f);
    float mean = sum / size;
    
    // Compute variance using functor
    float var_sum = thrust::transform_reduce(thrust::device_ptr<const float>(input), thrust::device_ptr<const float>(input + size), variance_op(mean), 0.0f, thrust::plus<float>());
    float variance = var_sum / size;
    
    // Apply normalization
    dim3 block(256);
    dim3 grid((size + block.x - 1) / block.x);
    instance_norm_kernel<<<grid, block>>>(input, output, mean, variance, size, eps);
}