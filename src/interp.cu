#include <cuda_runtime.h>
#include "interp.hpp"


__global__ void bilinear_interp_kernel(const FLOAT* input, FLOAT* output, int in_h, int in_w, int out_h, int out_w, int channels)
{
   int idx_c = threadIdx.x + blockIdx.x * blockDim.x;   // channel
   int out_y = threadIdx.y + blockIdx.y * blockDim.y;   // output row
   int out_x = threadIdx.z + blockIdx.z * blockDim.z;   // output col

   if (out_y < out_h && out_x < out_w && idx_c < channels)
   {
       // Map output coordinates to input coordinates. recheck this calculations
       float src_y = (float)(out_y + 0.5f) * in_h / out_h - 0.5f;
       float src_x = (float)(out_x + 0.5f) * in_w / out_w - 0.5f;

       // Clamp to valid range
       src_y = fmaxf(0.0f, fminf(src_y, in_h - 1.0f));
       src_x = fmaxf(0.0f, fminf(src_x, in_w - 1.0f));

       // Get integer and fractional parts
       int y1 = (int)floorf(src_y);
       int x1 = (int)floorf(src_x);
       int y2 = fminf(y1 + 1, in_h - 1);
       int x2 = fminf(x1 + 1, in_w - 1);

       float dy = src_y - y1;
       float dx = src_x - x1;

       // Get the 4 neighboring pixels
       int channel_offset = idx_c * in_h * in_w;
       float p1 = input[channel_offset + y1 * in_w + x1];  // top-left
       float p2 = input[channel_offset + y1 * in_w + x2];  // top-right
       float p3 = input[channel_offset + y2 * in_w + x1];  // bottom-left
       float p4 = input[channel_offset + y2 * in_w + x2];  // bottom-right

       // Bilinear interpolation
       float top    = p1 * (1.0f - dx) + p2 * dx;
       float bottom = p3 * (1.0f - dx) + p4 * dx;
       float result = top * (1.0f - dy) + bottom * dy;

       // Write to output
       int out_idx = idx_c * out_h * out_w + out_y * out_w + out_x;
       output[out_idx] = result;
   }
}

DevicePointer<FLOAT>& BilinearInterp2D::forward(const DevicePointer<FLOAT>& input_device)
{
   const int TC = 8;
   dim3 threadcount(TC, TC, TC);
   dim3 blocks((output_prop.channels + TC - 1) / TC,
               (output_prop.height   + TC - 1) / TC,
               (output_prop.width    + TC - 1) / TC);

   std::cout << "starting bilinear interp kernel " << input_prop << " -> " << output_prop 
             << " blocks: " << blocks.x << " " << blocks.y << " " << blocks.z << std::endl;

   bilinear_interp_kernel<<<blocks, threadcount>>>(input_device.get(), output_device.get(), input_prop.height, input_prop.width, output_prop.height, output_prop.width, input_prop.channels);
   
   cudaDeviceSynchronize();
   return output_device;
}

DevicePointer<FLOAT>& BilinearInterp2D::get_output()
{
    return output_device;
}

ImgProperty BilinearInterp2D::get_output_spec() const
{
    return output_prop;
}

ImgProperty BilinearInterp2D::get_input_spec() const
{
    return input_prop;
}

BilinearInterp2D::BilinearInterp2D(ImgProperty input_prop_, int target_height_, int target_width_) : input_prop(input_prop_), target_height(target_height_), target_width(target_width_)
{
   if (target_height <= 0 || target_width <= 0) 
       throw std::invalid_argument("target dimensions must be positive");

   output_prop.channels = input_prop.channels;  // Same channels
   output_prop.height   = target_height;
   output_prop.width    = target_width;

   std::vector<int> output_shape = {output_prop.channels, output_prop.height, output_prop.width};
   output_device.alloc(output_shape);
}
