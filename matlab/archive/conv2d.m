// Copyright 2025 Manoj Velmurugan
// SPDX-License-Identifier: MIT

function output = conv2d(input, kernel, k1, k2, s1, s2, p1, p2)
%CONV2D CUDA-accelerated 2D convolution
%   output = conv2d(input, kernel, k1, k2, s1, s2, p1, p2)
%
%   Inputs:
%     input  - Input tensor [ci, height, width] (single precision)
%     kernel - Convolution kernel [co, ci, k1, k2] (single precision)
%     k1, k2 - Kernel dimensions (must be odd)
%     s1, s2 - Stride in height/width direction (must be positive)
%     p1, p2 - Padding in height/width direction (must be non-negative)
%
%   Output:
%     output - Convolved tensor [co, output_height, output_width]

    % Input type validation
    if ~isa(input, 'single')
        error('Input must be single precision (use single())');
    end
    
    if ~isa(kernel, 'single')
        error('Kernel must be single precision (use single())');
    end
    
    % Input dimension validation
    if ndims(input) ~= 3
        error('Input must be 3D: [ci, height, width]. Got %dD', ndims(input));
    end
    
    if ndims(kernel) ~= 4
        error('Kernel must be 4D: [co, ci, k1, k2]. Got %dD', ndims(kernel));
    end
    
    % Get dimensions
    [ci, input_h, input_w] = size(input);
    [co, ci_kernel, k1_kernel, k2_kernel] = size(kernel);
    
    % Parameter validation
    if ~isscalar(k1) || ~isscalar(k2) || ~isscalar(s1) || ~isscalar(s2) || ~isscalar(p1) || ~isscalar(p2)
        error('k1, k2, s1, s2, p1, p2 must be scalar values');
    end
    
    % Convert to integers
    k1 = int32(k1); k2 = int32(k2);
    s1 = int32(s1); s2 = int32(s2);
    p1 = int32(p1); p2 = int32(p2);
    
    % Kernel size validation
    if mod(k1, 2) == 0 || mod(k2, 2) == 0
        error('Kernel dimensions k1=%d, k2=%d must be odd', k1, k2);
    end
    
    % Stride validation
    if s1 <= 0 || s2 <= 0
        error('Strides s1=%d, s2=%d must be positive', s1, s2);
    end
    
    % Padding validation
    if p1 < 0 || p2 < 0
        error('Padding p1=%d, p2=%d must be non-negative', p1, p2);
    end
    
    % Dimension consistency checks
    if ci ~= ci_kernel
        error('Input channels mismatch: input has %d channels, kernel expects %d', ci, ci_kernel);
    end
    
    if k1 ~= k1_kernel || k2 ~= k2_kernel
        error('Kernel size mismatch: parameters specify [%dx%d], kernel array is [%dx%d]', ...
              k1, k2, k1_kernel, k2_kernel);
    end
    
    % Output size validation (check for valid output dimensions)
    output_h = floor((input_h + 2*p1 - k1) / s1) + 1;
    output_w = floor((input_w + 2*p2 - k2) / s2) + 1;
    
    if output_h <= 0 || output_w <= 0
        error('Invalid output dimensions [%dx%d]. Check padding/stride/kernel size combination', ...
              output_h, output_w);
    end
    
    fprintf('Conv2D: [%d,%d,%d] * [%d,%d,%d,%d] -> [%d,%d,%d] (stride=[%d,%d], pad=[%d,%d])\n', ...
            ci, input_h, input_w, co, ci, k1, k2, co, output_h, output_w, s1, s2, p1, p2);
    
    % Call MEX function
    output = conv2d_mex(input, kernel, k1, k2, s1, s2, p1, p2);
end