#include <cuda_runtime.h>
#include "primitives.hpp"
#include <string>
#include <stdexcept>

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