#pragma once

#include "primitives.hpp"
#include <iostream>

class Fold2D
{
private:
    FLOAT *output_device; // can mean folded or unfolded. This class does not care what exactly is stored in the output side
    int height, width;
    const int reduction_ratio = 8;
    int channel_out = reduction_ratio * reduction_ratio;

public:
    Fold2D(int height_, int width_);
    ~Fold2D();
    
    FLOAT* fold(const FLOAT *input_device);
    FLOAT *unfold(const FLOAT *input_device);

    FLOAT *get_output();
};
