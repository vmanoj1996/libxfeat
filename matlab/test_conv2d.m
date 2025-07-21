% Create input data
clear mex
clear classes
clear all

% Apply a gassian blur on image using conv kernel

height = 200
width  = 400
 
% [ci, height, width]
input = single(randn(1, height, width));  

% [co, ci, k1, k2] 
kernel = single(ones(1, 1, 3, 3));
kernel = kernel / sum(abs(kernel(:)));

% Create structures for your C++ class

input_prop = clib.xfeat.ImgProperty(height, width)  % height, width
params     = clib.xfeat.Conv2DParams(3, 3, 1, 1, 1, 1, 1, 1)  % k1,k2,ci,co,s1,s2,p1,p2

conv = clib.xfeat.Conv2D(input_prop, params);
kernel_clib = flatten(kernel);

conv.set_kernel(kernel_clib);

% Input image processing
input_image = rgb2gray(imread('peppers.png'));
input_image = imresize(input_image, [height, width]);

input_device = host2device(input_image);
output_device = conv.forward(input_device);
output_host = device2host(output_device);

