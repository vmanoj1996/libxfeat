// Simple convolve 2d (non batched)

/*
Requirements:
1. No batch needed
2. preserve the size of the input image
3. input is  ci x H1 x W1
4. output is co x H2 x W2
5. Parameter count is co x ci x k1 x k2

Pytorch saves the parameters for conv layers in this format according to claude. Verify this
[out_channels, in_channels, kernel_height, kernel_width]

k1 for row and k2 for column
*/

#pragma once

#include <vector>
#include "primitives.hpp"
#include <memory>

#include "device_ops.hpp"

// careful while reordering
struct Conv2DParams {
   int k1, k2, ci, co;
   int s1, s2, p1, p2;
   
   Conv2DParams() = default;
   Conv2DParams(int k1_, int k2_, int ci_, int co_, int s1_, int s2_, int p1_, int p2_)
       : k1(k1_), k2(k2_), ci(ci_), co(co_), s1(s1_), s2(s2_), p1(p1_), p2(p2_) {}
};

inline std::ostream& operator<<(std::ostream& os, const Conv2DParams& params) {
   os << "Conv2DParams(k1=" << params.k1 << ", k2=" << params.k2 
      << ", ci=" << params.ci << ", co=" << params.co 
      << ", s1=" << params.s1 << ", s2=" << params.s2 
      << ", p1=" << params.p1 << ", p2=" << params.p2 << ")";
   return os;
}

template<typename Operation>
class Conv2D: public Layer
{

private:
    DevicePointer<FLOAT> kernel_device; // co, ci, k1, k2 order for cache optimality. Thats how pytorch is built too. optimized for row major operations
    DevicePointer<FLOAT> output_device; // co x output_height x output_width

    Conv2DParams params;
    ImgProperty input_prop, output_prop;

    // post operation
    Operation post_op;

public:
    Conv2D(ImgProperty input_prop_, Conv2DParams params_, const std::vector<FLOAT>& kernel_data, Operation post_op_); 
    Conv2D(ImgProperty input_prop_, Conv2DParams params_, const std::vector<FLOAT>& kernel_data): Conv2D(input_prop_, params_, kernel_data, Operation{}) {} // main constructor
    
    ~Conv2D(); // automatically made virtual by the compiler
    
    virtual const DevicePointer<FLOAT>& forward(const DevicePointer<FLOAT>& input_device);

    const DevicePointer<FLOAT>& get_output();
    Conv2DParams get_param() const;

    void set_kernel(const std::vector<FLOAT> &kernel_data);

    ImgProperty get_output_spec() const;
    ImgProperty get_input_spec()  const;

    void validate_params();


};
//  FACTORIES
inline std::unique_ptr<Layer> conv2d(ImgProperty input_prop, Conv2DParams params, const std::vector<FLOAT>& kernel_data) 
{
    return std::make_unique<Conv2D<Identity>>(input_prop, params, kernel_data);
}

template<typename Operation>
inline std::unique_ptr<Layer> conv2d(ImgProperty input_prop, Conv2DParams params, const std::vector<FLOAT>& kernel_data, Operation op) 
{
    return std::make_unique<Conv2D<Operation>>(input_prop, params, kernel_data, op);
}