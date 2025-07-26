
#pragma once

#include <vector>
#include "primitives.hpp"
#include <memory>

#include "device_ops.hpp"

struct PoolParams {
   int k1, k2; // kernel size in pooling
   int s1, s2, p1, p2;
   
   PoolParams() = default;
   PoolParams(int k1_, int k2_, int s1_, int s2_, int p1_ = 0, int p2_ = 0)
       : k1(k1_), k2(k2_), s1(s1_), s2(s2_), p1(p1_), p2(p2_) {}
};

class AvgPool2D: public Layer
{

private:
    DevicePointer<FLOAT> output_device; // co x output_height x output_width
    ImgProperty input_prop, output_prop;

    PoolParams params;

public:
    AvgPool2D(ImgProperty input_prop_, PoolParams params_); 
    ~AvgPool2D() = default; // automatically made virtual by the compiler
    
    virtual const DevicePointer<FLOAT>& forward(const DevicePointer<FLOAT>& input_device);
    const DevicePointer<FLOAT>& get_output();

    ImgProperty get_output_spec() const;
    ImgProperty get_input_spec()  const;
};