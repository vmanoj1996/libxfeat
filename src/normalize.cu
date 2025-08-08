
// Copyright 2025 Manoj Velmurugan
// SPDX-License-Identifier: MIT

#include <cub/cub.cuh>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/device_ptr.h>
#include "normalize.hpp"

__global__ void division_kernel(float* sum, int size) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        *sum = *sum / size;
    }
}

struct VarianceOp {
    float* mean_ptr;
    
    __host__ __device__ VarianceOp(float* mean) : mean_ptr(mean) {}
    
    __device__ float operator()(float x) const {
        float diff = x - *mean_ptr;
        return diff * diff;
    }
};

__global__ void instance_norm_kernel(const float* input, float* output, const float* mean, const float* variance, int size, float eps = 1e-5f)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < size) {
        float std_inv = rsqrtf(*variance + eps);
        output[idx] = (input[idx] - *mean) * std_inv;
    }
}

ImageNorm2D::ImageNorm2D(ImgProperty input_prop_, float eps_) : input_prop(input_prop_), eps(eps_), size(input_prop_.height * input_prop_.width)
{
    output_prop = {input_prop_.channels, input_prop_.height, input_prop_.width};

    std::vector<int> output_Shape = {output_prop.channels, output_prop.height, output_prop.width};
    output_device.alloc(output_Shape);
    setup_workspace();
}

ImageNorm2D::~ImageNorm2D() 
{
    cleanup();
}

void ImageNorm2D::setup_workspace() 
{
    // Determine CUB workspace size for sum reduction (same for both operations)
    cub::DeviceReduce::Sum(nullptr, temp_storage_bytes, (float*)nullptr, (float*)nullptr, size);
    
    // Allocate workspace
    cudaMalloc(&d_temp_storage, temp_storage_bytes);
    cudaMalloc(&d_sum_result, sizeof(float));
    cudaMalloc(&d_var_result, sizeof(float));
}

void ImageNorm2D::cleanup() 
{
    if (d_temp_storage) { cudaFree(d_temp_storage); d_temp_storage = nullptr; }
    if (d_sum_result) { cudaFree(d_sum_result); d_sum_result = nullptr; }
    if (d_var_result) { cudaFree(d_var_result); d_var_result = nullptr; }
}

DevicePointer<FLOAT>& ImageNorm2D::forward(const DevicePointer<FLOAT>& input_device) 
{
    // Compute sum using CUB
    cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, input_device.get(), d_sum_result, size);
    division_kernel<<<1, 1>>>(d_sum_result, size);
    cudaDeviceSynchronize();
    
    // Create thrust transform iterator for variance calculation
    VarianceOp variance_op(d_sum_result);
    thrust::device_ptr<const float> input_ptr(input_device.get()); // wrap the raw cuda device pointer to thrust
    auto variance_iter = thrust::make_transform_iterator(input_ptr, variance_op);
    
    // Compute variance using CUB Sum with thrust transform iterator
    cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, variance_iter, d_var_result, size);
    division_kernel<<<1, 1>>>(d_var_result, size);
    
    // Apply normalization
    dim3 block(256);
    dim3 grid((size + block.x - 1) / block.x);
    instance_norm_kernel<<<grid, block>>>(input_device.get(), output_device.get(), d_sum_result, d_var_result, size, eps);
    
    return output_device;
}