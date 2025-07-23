#include <cuda_runtime.h>
#include "primitives.hpp"
#include <string>
#include <stdexcept>
#include "device_ops.hpp"

// Common functionality to reuse 
template<typename T>
DevicePointer<T>::DevicePointer(int total_dim)
{
    alloc(total_dim);
}

template<typename T>
DevicePointer<T>::DevicePointer(const std::vector<T> &input, std::vector<int> dims_)
{
    alloc(dims_);
    set_value(input);
}

template<typename T>
DevicePointer<T>::~DevicePointer()
{
    if(ptr) cudaFree(ptr);
}

template<typename T>
T* DevicePointer<T>::get()
{
    return ptr;
}

template<typename T>
void DevicePointer<T>::alloc(std::vector<int> dims_)
{
    int total_dim = 1;
    for(auto dim: dims_)
    {
        total_dim *= dim;
    }
    dims = dims_;
    
    alloc(total_dim);
}

template<typename T>
void DevicePointer<T>::alloc(int total_dim)
{
    size = total_dim;
    if(dims.empty())
    {
        // If there is no multiple dims in vector, put the total dim instead
        dims.push_back(total_dim);
    }
    cudaMalloc(&ptr,    total_dim*sizeof(T));
    cudaMemset(&ptr, 0, total_dim*sizeof(T));
}

template<typename T>
void DevicePointer<T>::set_value(const std::vector<T> &input)
{
    if(ptr){
        cudaMemcpy(ptr, input.data(), input.size() * sizeof(T), cudaMemcpyHostToDevice);
    }
}


template<typename T>
std::vector<T> DevicePointer<T>::get_value() const
{
    std::vector<T> result(size);
    if(ptr && size > 0) {
        cudaError_t err = cudaMemcpy(result.data(), ptr, 
                                    size * sizeof(T), 
                                    cudaMemcpyDeviceToHost);
        if (err != cudaSuccess) {
            throw std::runtime_error("cudaMemcpy failed: " + std::string(cudaGetErrorString(err)));
        }
    }
    return result;
}

template<typename T>
std::vector<int> DevicePointer<T>::get_shape() const
{
    std::vector<int> result = dims;

    return result;
}


__global__ void init_BatchNormRelu_kernel(DeviceOp** p) 
{
    *p = new BatchNormRelu();
}

__global__ void setup_BatchNormRelu_kernel(DeviceOp* p, float* d_mean, float* d_var, int N) 
{
    BatchNormRelu* bn = static_cast<BatchNormRelu*>(p);
    bn->mean = d_mean;
    bn->var  = d_var;
    bn->N    = N;
}

__global__ void delete_op_kernel(DeviceOp** p) 
{
    if(*p)
    {
        (*p)->deleter();
        delete *p;
        *p = nullptr;
    }
}

DeviceOp* create_BatchNormRelu(const std::vector<float>& mean_host, 
                               const std::vector<float>& var_host) 
{
    // Validate input
    if (mean_host.size() != var_host.size()) {
        throw std::invalid_argument("Mean and var vectors must have same size");
    }
    
    int N = mean_host.size();
    
    // Allocate device memory for arrays
    float* d_mean;
    float* d_var;
    cudaMalloc(&d_mean, N * sizeof(float));
    cudaMalloc(&d_var,  N * sizeof(float));
    
    // Copy data to device
    cudaMemcpy(d_mean, mean_host.data(), N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_var, var_host.data(), N * sizeof(float), cudaMemcpyHostToDevice);
    
    // Just declare a device pointer - no allocation needed
    DeviceOp* d_post_op;
    
    // Create and setup the object (device new allocates the object)
    init_BatchNormRelu_kernel<<<1,1>>>(&d_post_op);
    setup_BatchNormRelu_kernel<<<1,1>>>(d_post_op, d_mean, d_var, N);
    cudaDeviceSynchronize();
    
    return d_post_op;
}