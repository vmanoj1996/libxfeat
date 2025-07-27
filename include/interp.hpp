// Copyright 2025 Manoj Velmurugan
// SPDX-License-Identifier: MIT


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
   
   virtual DevicePointer<FLOAT>& forward(const DevicePointer<FLOAT>& input_device);
   DevicePointer<FLOAT>& get_output();

   virtual ImgProperty get_output_spec() const {return output_prop;}
   virtual ImgProperty get_input_spec()  const {return input_prop;}
};

// Factory functions
inline std::unique_ptr<Layer> interp2d(ImgProperty input_prop, int height, int width) 
{
   return std::make_unique<BilinearInterp2D>(input_prop, height, width);
}
