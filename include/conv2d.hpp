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
#include "conv2d.hpp"

#include <cuda_runtime.h>
#include <cassert>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <memory>
#include <iostream>
#include <vector>
#include <assert.h>

#include <cuda/pipeline>
#include <cooperative_groups.h>

// careful while reordering
struct Conv2DParams {
   int k1, k2, ci, co;
   int s1, s2, p1, p2;
   
   Conv2DParams() = default;
   constexpr Conv2DParams(int k1_, int k2_, int ci_, int co_, int s1_, int s2_, int p1_, int p2_)
       : k1(k1_), k2(k2_), ci(ci_), co(co_), s1(s1_), s2(s2_), p1(p1_), p2(p2_) {}
};

inline std::ostream& operator<<(std::ostream& os, const Conv2DParams& params) {
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

public:
    Conv2D(ImgProperty input_prop_, const std::vector<FLOAT>& kernel_data, Operation post_op_, cudaStream_t stream_); 
    Conv2D(ImgProperty input_prop_, const std::vector<FLOAT>& kernel_data, cudaStream_t stream_): Conv2D(input_prop_, kernel_data, Operation{}, stream_) {}

    ~Conv2D(); // automatically made virtual by the compiler
    
    using Layer::forward;
    virtual DevicePointer<FLOAT>& forward(const DevicePointer<FLOAT>& input_device);

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

#ifdef __CUDACC__ // do not build the implementation for cpp files

template<Conv2DParams p, typename Operation>
__global__ void convolve2d_kernel(const FLOAT * __restrict__ input_device, const FLOAT * __restrict__ kernel_device, FLOAT * __restrict__ output_device, ImgProperty input_prop, ImgProperty output_prop, Operation op)
{
    // SETUP ------------------------------------------------------------------------------------------------------------------------
    extern __shared__ __align__(16) FLOAT kernel_per_ch[];

    const int idx_co  = threadIdx.x + blockIdx.x * blockDim.x; // channel output
    const int out_row = threadIdx.y + blockIdx.y * blockDim.y;
    const int out_col = threadIdx.z + blockIdx.z * blockDim.z;
    
    // cooperative copy to copy the kernel to shared memory
    const int tid = threadIdx.y * blockDim.z + threadIdx.z; // unique thread id (first channel id is 0 always)
    const int total_threads = blockDim.y * blockDim.z; 
    constexpr int kernel_net_size = p.ci * p.k1 * p.k2;

    // copy from idx_co * kernel_net_size to idx_co * kernel_net_size + kernel_net_size (last excluded)
    // first thread works on 0, 256, 512 ... till the end of kernel_net_size
    // second thread works on 1, 257, 513, ... till the end
    // last thread works on total_thread-1, total_thread-1-256 and so on. so no elements are skipped in this copy pattern.
    // plus all threads in the block get to do some work to copy our giant kernel.
    // plus all threads would work on closeby memory regions (which is cache friendly)

    //  this operation takes 50 uS
    for(int i = tid; i < kernel_net_size; i += total_threads) 
    {
        /* __ldg() is a CUDA intrinsic to perform read-only data cache loads from global memory.
            It tells the compiler and hardware that the data loaded won’t be written by any thread during kernel execution.
        */
        kernel_per_ch[i] = __ldg(&kernel_device[idx_co * kernel_net_size + i]);
    }

    // --------------------------------------------------------------------------------------------------------------------------------
    if (out_row >= output_prop.height || out_col >= output_prop.width || idx_co >= p.co) return;
        
    FLOAT sum = 0.0f;

    // once padded, the first operation that will happen is on this particular index in the imaginary padded input (implicit)
    const int in_row_start = out_row * p.s1 - p.p1;
    const int in_col_start = out_col * p.s2 - p.p2;

    __syncthreads(); // to sync the memory copy operations

    #pragma unroll
    for (int idx_ci = 0; idx_ci < p.ci; idx_ci++)
    {
        const int input_base = idx_ci * input_prop.height * input_prop.width + in_row_start * input_prop.width;
        const int kernel_base = idx_ci * (p.k1 * p.k2);

        #pragma unroll
        for (int kernel_row = 0; kernel_row < p.k1; kernel_row++)
        {
            const int input_row_offset = kernel_row * input_prop.width;
            const int kernel_row_offset = kernel_row * p.k2;

            const int input_row_index = (in_row_start + kernel_row);
            const bool row_valid = (input_row_index >= 0 && input_row_index < input_prop.height);

            #pragma unroll
            for (int kernel_col = 0; kernel_col < p.k2; kernel_col++)
            {                    
                // load kernel from shared memory for faster access
                FLOAT kernel_value = kernel_per_ch[kernel_base + kernel_row_offset + kernel_col];

                const int input_col_index = (in_col_start + kernel_col);
                const bool col_valid = (input_col_index >= 0 && input_col_index < input_prop.width);

                // FLOAT input_value = (row_valid && col_valid)? input_device[idx_ci * input_prop.height * input_prop.width + input_row_index * input_prop.width + input_col_index]: 0.0f;
                const FLOAT input_value = (row_valid && col_valid) ? input_device[input_base + input_row_offset + input_col_index] : 0.0f;
                sum += input_value * kernel_value;
            }
        }
    }

    const int o_index = idx_co * output_prop.height * output_prop.width + out_row * output_prop.width + out_col;

    output_device[o_index] = op.forward(sum, idx_co);
}

template<Conv2DParams params, typename Operation>
Conv2D<params, Operation>::Conv2D(ImgProperty input_prop_, const std::vector<FLOAT> &kernel_data, Operation post_op_, cudaStream_t stream_):  input_prop(input_prop_), post_op(post_op_)
{
    stream = stream_;
    
    if (params.k1 % 2 == 0)
    {
        throw std::invalid_argument("k1 must be odd");
    }
    if (params.k2 % 2 == 0)
    {
        throw std::invalid_argument("k2 must be odd");
    }
    if (params.s1 <= 0)
    {
        throw std::invalid_argument("s1 (stride height) must be positive");
    }
    if (params.s2 <= 0)
    {
        throw std::invalid_argument("s2 (stride width) must be positive");
    }
    if (params.p1 < 0)
    {
        throw std::invalid_argument("p1 (padding height) must be non-negative");
    }
    if (params.p2 < 0)
    {
        throw std::invalid_argument("p2 (padding width) must be non-negative");
    }

    output_prop.channels = params.co;
    output_prop.height = (input_prop.height + 2 * params.p1 - params.k1) / params.s1 + 1;
    output_prop.width =  (input_prop.width + 2 * params.p2 - params.k2) / params.s2 + 1;

    std::vector<int> output_Shape = {params.co, output_prop.height, output_prop.width};
    output_device.alloc(output_Shape);

    std::vector<int> kernel_Shape = {params.co, params.ci, params.k1, params.k2};
    kernel_device.alloc(kernel_Shape);

    // set the kernel up
    set_kernel(kernel_data);
}

template<Conv2DParams params, typename Operation>
Conv2D<params, Operation>::~Conv2D()
{
    post_op.destroy();
}

template<Conv2DParams params, typename Operation>
DevicePointer<FLOAT> &Conv2D<params, Operation>::forward(const DevicePointer<FLOAT> &input_device)
{
    std::vector<int> expected_shape = {input_prop.channels, input_prop.height, input_prop.width};
    auto actual_shape = input_device.get_shape();

    if (actual_shape != expected_shape) throw std::runtime_error("conv2d: shape mismatch");

    // spawn just 1 thread per output channel. but it is still effective. the 3d indexing is just for convenience. 
    // each exec block would correspond to 16x16 output tile for a single output channel

    // This happens to be the most optimal and launchable kernel. two warps 32, 32 will access memory regions close to each other. horizontal scanning would be encouraged which is cache friendly I guess
    dim3 threadcount(1, 2, 64); 
    dim3 blocks(params.co, (output_prop.height + threadcount.y - 1) / threadcount.y, (output_prop.width + threadcount.z - 1) / threadcount.z);

    const int kernel_shared_per_block = params.k1 * params.k2 * params.ci * sizeof(FLOAT);
    const int kernel_shared_per_block_padded = ((kernel_shared_per_block + 15) / 16) * 16; // 16 byte padding.

#ifdef ENABLE_XFEAT_DEBUG
    std::cout<<"starting conv kernel "<<input_prop<<" "<<output_prop<<" "<<params<<blocks.x<<" "<<blocks.y<<" "<<blocks.z<<" "<<std::endl;
#endif

    convolve2d_kernel<params><<<blocks, threadcount, kernel_shared_per_block_padded, stream>>>(input_device.get(), kernel_device.get(), output_device.get(), input_prop, output_prop, post_op);

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