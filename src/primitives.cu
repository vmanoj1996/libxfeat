#include <cuda_runtime.h>
#include "primitives.hpp"
#include <string>
#include <stdexcept>
#include "device_ops.hpp"

// Common functionality to reuse 
// template<typename T>
// DevicePointer<T>::DevicePointer(int total_dim)
// {
//     alloc(total_dim);
// }

template<typename T>
DevicePointer<T>::DevicePointer(const std::vector<T> &input, std::vector<int> dims_)
{
    alloc(dims_);
    set_value(input);
}

template<typename T>
DevicePointer<T>::DevicePointer(const DevicePointer<T> &input)
{
    // copy constructor
    alloc(input.dims);
    cudaMemcpy(ptr, input.get(), input.size * sizeof(T), cudaMemcpyDeviceToDevice);
}

template<typename T>
DevicePointer<T>::~DevicePointer()
{
    if(ptr) cudaFree(ptr);
}

template<typename T>
T* DevicePointer<T>::get() const
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

    if(ptr) 
    {
        throw std::runtime_error("ptr already allocated\n");
    }

    cudaError_t result = cudaMalloc(&ptr,    total_dim*sizeof(T));
    cudaMemset(ptr, 0, total_dim*sizeof(T));

    if (result != cudaSuccess) {
        std::string error_msg = "CUDA malloc failed: " + std::string(cudaGetErrorString(result)) + 
                               "\nCall stack:\n" + boost::stacktrace::to_string(boost::stacktrace::stacktrace());
        throw std::runtime_error(error_msg);
    }
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
        cudaError_t err = cudaMemcpy(result.data(), ptr, size * sizeof(T), cudaMemcpyDeviceToHost);
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

template<typename T>
void DevicePointer<T>::print_shape() const
{
    for(auto dim: dims)
    {
        std::cout<<dim<<" ";
    }
    std::cout<<"\n";
}
