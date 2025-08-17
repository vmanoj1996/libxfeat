// Copyright 2025 Manoj Velmurugan
// SPDX-License-Identifier: MIT

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

#include "primitives.hpp"
#include "layer.hpp"
#include "device_ops.hpp"

#include <cuda_runtime.h>
#include <cassert>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <memory>
#include <iostream>
#include <vector>
#include <assert.h>
#include <type_traits>
#include <fstream>

#include <cuda/pipeline>
#include <cooperative_groups.h>

// careful while reordering
struct Conv2DParams 
{
   int k1, k2, ci, co;
   int s1, s2, p1, p2;
   
   Conv2DParams() = default;
   constexpr Conv2DParams(int k1_, int k2_, int ci_, int co_, int s1_, int s2_, int p1_, int p2_)
       : k1(k1_), k2(k2_), ci(ci_), co(co_), s1(s1_), s2(s2_), p1(p1_), p2(p2_) {}
};


inline std::ostream& operator<<(std::ostream& os, const Conv2DParams& params) 
{
   os << "Conv2DParams(k1=" << params.k1 << ", k2=" << params.k2 
      << ", ci=" << params.ci << ", co=" << params.co 
      << ", s1=" << params.s1 << ", s2=" << params.s2 
      << ", p1=" << params.p1 << ", p2=" << params.p2 << ")";
   return os;
}

template<Conv2DParams params, typename Operation>
class Conv2D: public Layer
{

private:
    DevicePointer<FLOAT> kernel_device; // co, ci, k1, k2 order for cache optimality. Thats how pytorch is built too. optimized for row major operations
    DevicePointer<FLOAT> output_device; // co x output_height x output_width

    // Conv2DParams params;
    ImgProperty input_prop, output_prop;

    // post operation
    Operation post_op;

    // input converted to row
    DevicePointer<FLOAT> input_row;
    int input_M;
    int input_N;

    // profile guided threadcount storage
    dim3 tc_im2col{4, 32}; // this is pretty close to the optimal
    dim3 get_profiled_threadcount();

public:
    Conv2D(ImgProperty input_prop_, const std::vector<FLOAT>& kernel_data, Operation post_op_, cudaStream_t stream_); 
    Conv2D(ImgProperty input_prop_, const std::vector<FLOAT>& kernel_data, cudaStream_t stream_): Conv2D(input_prop_, kernel_data, Operation{}, stream_) {}

    ~Conv2D(); // automatically made virtual by the compiler
    
    using Layer::forward;
    virtual DevicePointer<FLOAT>& forward(const DevicePointer<FLOAT>& input_device);
    DevicePointer<FLOAT>& forward_profile(const DevicePointer<FLOAT>& input_device, int tc1=0, int tc2=0);

    DevicePointer<FLOAT>& get_output();
    Conv2DParams get_param() const;

    void set_kernel(const std::vector<FLOAT> &kernel_data);

    virtual ImgProperty get_output_spec() const {return output_prop;}
    virtual ImgProperty get_input_spec()  const {return input_prop;}

};

//  FACTORIES
template<Conv2DParams params, typename Operation>
inline std::unique_ptr<Layer> conv2d(ImgProperty input_prop, const std::vector<FLOAT>& kernel_data, Operation op, cudaStream_t stream_) 
{
    return std::make_unique<Conv2D<params, Operation>>(input_prop, kernel_data, op, stream_);
}

// KERNEL  --------------------------------------------------------------------------------------------------------------------------------------------
#ifdef __CUDACC__ // do not build the implementation for cpp files. only cu files will build this section

inline __global__ void im2col_kernel(const FLOAT __restrict__ *input, FLOAT __restrict__ *output, Conv2DParams p, ImgProperty iprop, ImgProperty oprop, int M, int N)
{
    // output is M x N - M number of patches and N is the patch size.
    const int m = threadIdx.x + blockDim.x * blockIdx.x;
    const int n = threadIdx.y + blockDim.y * blockIdx.y;

    const int k1k2 = p.k1*p.k2;

    // guard for the last warp
    if(m>=M || n>=N) return;

    // refer to my (manoj) notes to check the conventions and symbol meanings
    // get the dimensions corresponding to the output row and column
    int beta_o  = m / oprop.width;
    int gamma_o = m % oprop.width;

    // get the top left corner of the input patch. basically the row and column
    int beta_i_  = p.s1 * beta_o  - p.p1; 
    int gamma_i_ = p.s2 * gamma_o - p.p2;

    // get the patch channel and patch local row, col
    int alpha_i = n / k1k2;
    int theta  = (n % k1k2) / p.k2;
    int phi    = (n % k1k2) % p.k2;

    // get the row and column of the input corresponding to m, n
    int beta_i  = beta_i_  + theta;
    int gamma_i = gamma_i_ + phi;

    // gather operation
    bool valid = (beta_i>=0 && beta_i<iprop.height && gamma_i>=0 && gamma_i<iprop.width);
    output[m*N + n] = valid ? input[alpha_i*(iprop.height*iprop.width) + beta_i*iprop.width + gamma_i]:0.0f;
}

// IMPLEMENTATION ------------------------------------------------------------------------------------------------------------------------------------------------------------------------
template<Conv2DParams params, typename Operation>
inline dim3 Conv2D<params, Operation>::get_profiled_threadcount() 
{
    // hardcoded too much including the image dimensions lol. 
    // This is a vanilla profile guided optimization

    // Structure to hold profile entry
    struct ProfileEntry {
        int k, ci, co, s, p, h, w, tc1, tc2;
    };
    
    // Hardcoded best configurations from your profiling results
    static const ProfileEntry profiles[] = {
        {3, 1, 4, 1, 1, 480, 640, 8, 16},
        {3, 4, 8, 2, 1, 480, 640, 4, 64},
        {3, 8, 8, 1, 1, 240, 320, 4, 32},
        {3, 8, 24, 2, 1, 240, 320, 4, 32},
        {1, 1, 24, 1, 0, 120, 160, 128, 1},
        {3, 24, 24, 1, 1, 120, 160, 4, 32},
        {3, 24, 64, 2, 1, 120, 160, 4, 32},
        {3, 64, 64, 1, 1, 60, 80, 4, 32},
        {1, 64, 64, 1, 0, 60, 80, 4, 16},
        {3, 64, 64, 2, 1, 60, 80, 4, 32},
        {3, 64, 64, 1, 1, 30, 40, 4, 32},
        {3, 64, 128, 2, 1, 30, 40, 4, 32},
        {3, 128, 128, 1, 1, 15, 20, 4, 64},
        {1, 128, 64, 1, 0, 15, 20, 4, 32},
        {1, 64, 1, 1, 0, 60, 80, 4, 16},
        {1, 64, 65, 1, 0, 60, 80, 4, 16}
    };
    
    // Search for matching configuration using member variables and template params
    for (const auto& entry : profiles) {
        if (entry.k == params.k1 && entry.ci == params.ci && entry.co == params.co && 
            entry.s == params.s1 && entry.p == params.p1 && 
            entry.h == input_prop.height && entry.w == input_prop.width) {
            return dim3(entry.tc1, entry.tc2);
        }
    }
    
    // Default if not found
    return dim3(4, 32);
}

template<Conv2DParams params, typename Operation>
Conv2D<params, Operation>::Conv2D(ImgProperty input_prop_, const std::vector<FLOAT> &kernel_data, Operation post_op_, cudaStream_t stream_):  input_prop(input_prop_), post_op(post_op_)
{
    stream = stream_;
    
    static_assert(params.k1 % 2 == 1, "k1 must be odd");
    static_assert(params.k2 % 2 == 1, "k2 must be odd"); 
    static_assert(params.s1 > 0, "s1 must be positive");
    static_assert(params.s2 > 0, "s2 must be positive");
    static_assert(params.p1 >= 0, "p1 must be non-negative");
    static_assert(params.p2 >= 0, "p2 must be non-negative");

    output_prop.channels = params.co;
    output_prop.height = (input_prop.height + 2 * params.p1 - params.k1) / params.s1 + 1;
    output_prop.width =  (input_prop.width + 2 * params.p2 - params.k2) / params.s2 + 1;

    std::vector<int> output_Shape = {params.co, output_prop.height, output_prop.width};
    output_device.alloc(output_Shape);

    std::vector<int> kernel_Shape = {params.co, params.ci, params.k1, params.k2};
    kernel_device.alloc(kernel_Shape);

    // set the kernel up
    set_kernel(kernel_data);

    // set the img2row matrix (input in row form)
    input_M = output_prop.height *output_prop.width;
    input_N = params.ci*params.k1*params.k2;
    std::vector<int> input_row_shape = {input_M, input_N};
    input_row.alloc(input_row_shape);

    // set the optimized kernel launch parameters obtained with pgo
    tc_im2col = get_profiled_threadcount();

}

template<Conv2DParams params, typename Operation>
Conv2D<params, Operation>::~Conv2D()
{
    post_op.destroy();
}

// #define CONV_PERF_TUNE

template<Conv2DParams params, typename Operation>
DevicePointer<FLOAT> &Conv2D<params, Operation>::forward_profile(const DevicePointer<FLOAT> &input_device, int tc1, int tc2)
{
    std::vector<int> expected_shape = {input_prop.channels, input_prop.height, input_prop.width};
    auto actual_shape = input_device.get_shape();

    if (actual_shape != expected_shape) throw std::runtime_error("conv2d: shape mismatch");

    if(tc1==0 && tc2 ==0) tc_im2col = dim3(2, 32); 
    else tc_im2col = dim3(tc1, tc2); 

    dim3 blocks((input_M + tc_im2col.x - 1)/tc_im2col.x, (input_N + tc_im2col.y - 1)/tc_im2col.y);

    im2col_kernel<<<blocks, tc_im2col, 0, stream>>>(input_device.get(), input_row.get(), params, input_prop, output_prop, input_M, input_N);
    
    CUDA_SYNC_IF_NEEDED();

    return output_device;
}

template<Conv2DParams params, typename Operation>
DevicePointer<FLOAT> &Conv2D<params, Operation>::forward(const DevicePointer<FLOAT> &input_device)
{

// #ifdef CONV_PERF_TUNE
//     // Save this layer's configuration to file
//     // delete the txt file before launch or u get two files.
//     std::ofstream config_file("conv2d_layer_configs.txt" , std::ios::app);
//     config_file << params.k1 << " " << params.k2 << " " << params.ci << " " << params.co << " "
//                 << params.s1 << " " << params.s2 << " " << params.p1 << " " << params.p2 << " "
//                 << input_prop.channels << " " << input_prop.height << " " << input_prop.width << "\n";
//     config_file.close();
// #endif

    std::vector<int> expected_shape = {input_prop.channels, input_prop.height, input_prop.width};
    auto actual_shape = input_device.get_shape();

    if (actual_shape != expected_shape) throw std::runtime_error("conv2d: shape mismatch");

    dim3 blocks((input_M + tc_im2col.x - 1)/tc_im2col.x, (input_N + tc_im2col.y - 1)/tc_im2col.y);

    im2col_kernel<<<blocks, tc_im2col, 0, stream>>>(input_device.get(), input_row.get(), params, input_prop, output_prop, input_M, input_N);
    
    CUDA_SYNC_IF_NEEDED();

    return output_device;
}

template<Conv2DParams params, typename Operation>
void Conv2D<params, Operation>::set_kernel(const std::vector<FLOAT> &kernel_data)
{
    size_t expected_size = params.co * params.ci * params.k1 * params.k2;
    if (kernel_data.size() != expected_size)
    {
        throw std::invalid_argument("Kernel size mismatch: expected " + std::to_string(expected_size) + " weights, got " + std::to_string(kernel_data.size()));
    }
    cudaMemcpy(kernel_device.get(), kernel_data.data(), expected_size * sizeof(FLOAT), cudaMemcpyHostToDevice);
}

template<Conv2DParams params, typename Operation>
DevicePointer<FLOAT> &Conv2D<params, Operation>::get_output()
{
    return output_device;
}

template<Conv2DParams params, typename Operation>
Conv2DParams Conv2D<params, Operation>::get_param() const
{
    return params;
}

#endif // __CUDACC__