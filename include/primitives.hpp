#pragma once
#include<vector>

using FLOAT = float;
// #define FLOAT float

struct ImgProperty
{
    int channels; // used only in few places
    int height;
    int width;

    ImgProperty() = default;
    ImgProperty(int height_, int width_): height(height_), width(width_){}
    ImgProperty(int channels_, int height_, int width_): channels(channels_), height(height_), width(width_){}
};

template<typename T>
class DevicePointer
{
    private:
    T *ptr;
    size_t size; // total size
    std::vector<int> dims;
    
    public:
    DevicePointer() = default;
    DevicePointer(int total_dim);
    DevicePointer(const std::vector<T> &input, std::vector<int> dims_);
    ~DevicePointer();

    T* get() const;
    
    void alloc(std::vector<int> dims_);
    void alloc(int total_dim);

    void set_value(const std::vector<T> &input);
    std::vector<T> get_value() const;
    std::vector<int> get_shape() const;
    // const means it wont modify the class object in any way

    // Disable copy/move operations
    DevicePointer(const DevicePointer&) = delete;
    DevicePointer(DevicePointer&&) = delete;
    DevicePointer& operator=(const DevicePointer&) = delete;
    DevicePointer& operator=(DevicePointer&&) = delete;
};

// forward declarations of all potential combinations 
template class DevicePointer<FLOAT>;
template class DevicePointer<int>;


class Layer
{
    public:
    virtual const DevicePointer<FLOAT>& forward(const DevicePointer<FLOAT>& input_device) = 0;

    virtual ~Layer() = default;
};