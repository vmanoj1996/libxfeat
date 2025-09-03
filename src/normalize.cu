
// Copyright 2025 Manoj Velmurugan
// SPDX-License-Identifier: MIT

#include <cub/cub.cuh>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/device_ptr.h>
#include "normalize.hpp"

#define EPS 1e-5f

__global__ void division_kernel(float* sum, int size) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        *sum = *sum / size;
    }
}

struct VarianceOp {
    float* mean_ptr;
    
    __host__ __device__ inline VarianceOp(float* mean) : mean_ptr(mean) {}
    
    __device__ inline float operator()(float x) const {
        float diff = x - *mean_ptr;
        return diff * diff;
    }
};

__global__ void instance_norm_kernel(const float* __restrict__ input, float* __restrict__ output, const float* __restrict__ mean, const float* __restrict__ variance, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx >= size) return;

    float std_inv = rsqrtf(*variance + EPS);
    output[idx] = (input[idx] - *mean) * std_inv;
    
}

ImageNorm2D::ImageNorm2D(ImgProperty input_prop_, float eps_, cudaStream_t stream_) : input_prop(input_prop_), size(input_prop_.height * input_prop_.width)
{
    stream = stream_;
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
    cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, input_device.get(), d_sum_result, size, stream);
    division_kernel<<<1, 1, 0, stream>>>(d_sum_result, size);
    CUDA_SYNC_IF_NEEDED();
    
    // Create thrust transform iterator for variance calculation
    VarianceOp variance_op(d_sum_result);
    thrust::device_ptr<const float> input_ptr(input_device.get()); // wrap the raw cuda device pointer to thrust
    auto variance_iter = thrust::make_transform_iterator(input_ptr, variance_op);
    
    // Compute variance using CUB Sum with thrust transform iterator
    cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, variance_iter, d_var_result, size, stream);
    division_kernel<<<1, 1, 0, stream>>>(d_var_result, size);
    
    // Apply normalization
    dim3 block(256);
    dim3 grid((size + block.x - 1) / block.x);
    instance_norm_kernel<<<grid, block, 0, stream>>>(input_device.get(), output_device.get(), d_sum_result, d_var_result, size);
    
    return output_device;
}