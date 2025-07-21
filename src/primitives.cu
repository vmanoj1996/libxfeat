#include <cuda_runtime.h>
#include "primitives.hpp"

// Common functionality to reuse 
template<typename T>
DevicePointer<T>::DevicePointer()
{
    
}

template<typename T>
DevicePointer<T>::~DevicePointer()
{
    if(ptr) cudaFree(ptr);
}

template<typename T>
T* DevicePointer<T>::get() const
{
    return T;
}

template<typename T>
void DevicePointer<T>::alloc(int total_dim)
{
    cudaMalloc(&ptr,    total_dim*sizeof(T));
    cudaMemset(&ptr, 0, total_dim*sizeof(T));
}