function col_major_data = r2c(row_major_data)
% Convert N-dimensional data from row-major to column-major layout
% 
% Inputs:
%   row_major_data: 1D vector containing data in row-major order
%   original_shape: vector of dimensions [dim1, dim2, ..., dimN]
%
% Output:
%   col_major_data: reshaped data in column-major order

    % For N-D arrays, MATLAB's permute with reversed dimension order
    % effectively converts row-major to column-major
    
    % Reshape the 1D data to N-D using row-major interpretation
    % We need to reverse the shape and then permute

    original_shape = fliplr(size(row_major_data));

    reversed_shape = fliplr(original_shape);
    
    % Reshape using reversed dimensions (this mimics row-major filling)
    temp_data = reshape(row_major_data, reversed_shape);
    
    % Create permutation order to flip all dimensions
    ndims = length(original_shape);
    perm_order = ndims:-1:1;
    
    % Permute to get correct column-major layout
    col_major_data = permute(temp_data, perm_order);
end