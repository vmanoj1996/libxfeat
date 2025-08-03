// Copyright 2025 Manoj Velmurugan
// SPDX-License-Identifier: MIT

#pragma once

#include "primitives.hpp"
#include <memory>
#include <vector>

class Add : public Layer
{
private:
    DevicePointer<FLOAT> output_device;
    ImgProperty output_prop, input_prop;

public:
    Add(ImgProperty output_prop_);
    ~Add() = default;
    
    // Forward for multiple inputs
    virtual DevicePointer<FLOAT>& forward(const std::vector<const DevicePointer<FLOAT>*>& inputs) override;
    virtual DevicePointer<FLOAT>& forward(const DevicePointer<FLOAT>& input) override;
    
    virtual ImgProperty get_output_spec() const {return output_prop;}
    virtual ImgProperty get_input_spec()  const {return input_prop;}
};

// Factory function
inline std::unique_ptr<Layer> add_layer(ImgProperty output_prop) 
{
    return std::make_unique<Add>(output_prop);
}