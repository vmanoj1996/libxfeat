
#pragma once

#include <vector>
#include "primitives.hpp"
#include <memory>

#include "device_ops.hpp"

class BilinearInterp2D: public Layer
{
private:
   DevicePointer<FLOAT> output_device;
   ImgProperty input_prop, output_prop;

   int target_height, target_width;

public:
   BilinearInterp2D(ImgProperty input_prop_, int target_height_, int target_width_); 
   ~BilinearInterp2D() = default;
   
   virtual const DevicePointer<FLOAT>& forward(const DevicePointer<FLOAT>& input_device);
   const DevicePointer<FLOAT>& get_output();

   ImgProperty get_output_spec() const;
   ImgProperty get_input_spec() const;
};

// Factory functions
inline std::unique_ptr<Layer> interp2d(ImgProperty input_prop, int height, int width) 
{
   return std::make_unique<BilinearInterp2D>(input_prop, height, width);
}


inline const DevicePointer<FLOAT>& BilinearInterp2D::get_output()
{
    return output_device;
}

inline ImgProperty BilinearInterp2D::get_output_spec() const
{
    return output_prop;
}

inline ImgProperty BilinearInterp2D::get_input_spec() const
{
    return input_prop;
}

inline BilinearInterp2D::BilinearInterp2D(ImgProperty input_prop_, int target_height_, int target_width_) : input_prop(input_prop_), target_height(target_height_), target_width(target_width_)
{
   if (target_height <= 0 || target_width <= 0) 
       throw std::invalid_argument("target dimensions must be positive");

   output_prop.channels = input_prop.channels;  // Same channels
   output_prop.height   = target_height;
   output_prop.width    = target_width;

   std::vector<int> output_shape = {output_prop.channels, output_prop.height, output_prop.width};
   output_device.alloc(output_shape);
}
