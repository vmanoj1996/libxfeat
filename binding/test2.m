% Create input data
input = single(randn(3, 32, 32));    % [ci, height, width]
kernel = single(randn(16, 3, 3, 3)); % [co, ci, k1, k2]

% Create structures for your C++ class
input_prop = clib.conv2d.ImgProperty(32, 32);  % height, width
params = clib.conv2d.Conv2DParams(3, 3, 3, 16, 1, 1, 1, 1);  % k1,k2,ci,co,s1,s2,p1,p2

% Create convolution object
conv = clib.conv2d.Convolve2D(input_prop, params);

% Set kernel weights (need to flatten to 1D vector)
kernel_flat = kernel(:);  % Flatten 4D -> 1D
conv.set_kernel(kernel_flat);