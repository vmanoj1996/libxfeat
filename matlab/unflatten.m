function output_matrix = unflatten(flat_array, target_shape)
   % UNFLATTEN Convert 1D clib array back to MATLAB matrix with proper shape
   % Handles C row-major to MATLAB column-major conversion
   %
   % Inputs:
   %   flat_array   - 1D clib array (1xN) from C++
   %   target_shape - Original MATLAB matrix dimensions [d1, d2, d3, ...]
   %
   % Output:
   %   output_matrix - Reshaped MATLAB matrix in column-major order
   
   % Convert clib array to MATLAB array
   flat_data = single(flat_array);  % or double(flat_array)
   
   % Ensure it's a column vector for reshaping
   flat_data = flat_data(:);
   
   % Reshape using reversed dimensions (since data is in C row-major order)
   reversed_shape = target_shape(end:-1:1);
   reshaped_data = reshape(flat_data, reversed_shape);
   
   % Permute back to MATLAB column-major order
   output_matrix = permute(reshaped_data, ndims(reshaped_data):-1:1);
end