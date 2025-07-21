#pragma once

using FLOAT = float;

struct ImgProperty
{
    int height;
    int width;
};

template<typename T>
class DevicePointer
{
    private:
    T *ptr;

    public:
    DevicePointer();
    ~DevicePointer();

    T* get();
    void alloc(int total_dim);

    // Disable copy/move operations
    DevicePointer(const DevicePointer&) = delete;
    DevicePointer(DevicePointer&&) = delete;
    DevicePointer& operator=(const DevicePointer&) = delete;
    DevicePointer& operator=(DevicePointer&&) = delete;
};

// forward declarations of all potential combinations 
template class DevicePointer<FLOAT>;
template class DevicePointer<int>;
