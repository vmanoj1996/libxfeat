// Copyright 2025 Manoj Velmurugan
// SPDX-License-Identifier: MIT

function flatarray = flatten(input_matrix)
% FLATTEN convert matlab column major to c-major and flatten. return a clib array that maps to c++ vector

input_reordered = permute(input_matrix, ndims(input_matrix):-1:1);  % Reverse all dimensions
input_flat = input_reordered(:);

size(input_flat)

% 1 x N gets treated as 1D array
flatarray = clibConvertArray("clib.xfeat.Float", single(input_flat'));

% Check if creation worked. It should have only one dimension to correctly map to vector<float> on cpp
% kernel_flat_clib.Dimensions

% https://www.mathworks.com/help/matlab/matlab_external/pass-stdvector-parameter.html

% help test_conv2d
% --- help for clib.xfeat.Conv2D/set_kernel ---

% set_kernel -  set_kernel Method of C++ class Conv2D.

%   clib.xfeat.Conv2D/set_kernel(obj,kernel_data)
%     Input Arguments
%       obj            read-only clib.xfeat.Conv2D  
%       kernel_data    vector read-only clib.array.xfeat.Float  


end