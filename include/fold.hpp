// Copyright 2025 Manoj Velmurugan
// SPDX-License-Identifier: MIT

#pragma once

#include "primitives.hpp"
#include <iostream>
#include <memory>


class Fold2D_common: public Layer
{
protected:
    DevicePointer<FLOAT> output_device;
    const int reduction_ratio = 8;

    int height, width;
    ImgProperty input_prop, output_prop; 

    public:
    Fold2D_common(int height_, int width_): height(height_), width(width_) {}
    ~Fold2D_common() = default; // this will destroy the output_device

    virtual ImgProperty get_output_spec() const {return output_prop;}
    virtual ImgProperty get_input_spec()  const {return input_prop;}

};


class Fold2D: public Fold2D_common
{
private:

public:
    Fold2D(int height_, int width_);

    virtual DevicePointer<FLOAT>& forward(const DevicePointer<FLOAT>& input_device);

};


class UnFold2D: public Fold2D_common
{
private:

public:
    UnFold2D(int height_, int width_);
    
    virtual DevicePointer<FLOAT>& forward(const DevicePointer<FLOAT>& input_device);
};


template<typename... Args>
std::unique_ptr<Layer> make_fold(Args&&... args) {
    return std::make_unique<Fold2D>(std::forward<Args>(args)...);
}

template<typename... Args>
std::unique_ptr<Layer> make_unfold(Args&&... args) {
    return std::make_unique<UnFold2D>(std::forward<Args>(args)...);
}