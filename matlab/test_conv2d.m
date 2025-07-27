// Copyright 2025 Manoj Velmurugan
// SPDX-License-Identifier: MIT

% Create input data
clear mex
clear classes
clear all

height = 200
width  = 400
input_prop = clib.xfeat.ImgProperty(height, width)  % height, width

input_image = rgb2gray(imread('peppers.png'));
input_image = imresize(input_image, [height, width]);
input_device = host2device(input_image);

%% Apply a gaussian blur on image using conv kernel
% [co, ci, k1, k2] 
kernel = single(ones(1, 1, 3, 3));
kernel = kernel / sum(abs(kernel(:)));
params     = clib.xfeat.Conv2DParams(3, 3, 1, 1, 1, 1, 1, 1)  % k1,k2,ci,co,s1,s2,p1,p2

blur = clib.xfeat.Conv2D(input_prop, params);
kernel_clib = flatten(kernel);
blur.set_kernel(kernel_clib);

output_device = blur.forward(input_device);
output_host = device2host(output_device);


%% Apply Sobel edge detection horizontally
sobel_x = single([-1, 0, 1; 
                 -2, 0, 2; 
                 -1, 0, 1]);

sobel_filter = clib.xfeat.Conv2D(input_prop, params);

kernel_clib = flatten(sobel_x);
sobel_filter.set_kernel(kernel_clib);

output_device = sobel_filter.forward(input_device);
sobel_output = device2host(output_device);

% sobel also worked

%% Lets have two images batched. apply gaussian on first channel and sobel only on second

input_image2 = checkerboard(20, 5, 10);
input_image2 = imresize(input_image2, [height, width]);
input_image2 = im2single(input_image2);

% Create 2-channel input [ci=2, height, width]
input_2ch = zeros(2, height, width, 'single');
input_2ch(1, :, :) = input_image;
input_2ch(2, :, :) = input_image2; 

kernel_2ch = zeros(2, 2, 3, 3, 'single');

gaussian = single(ones(3, 3)) / 9;
kernel_2ch(1, 1, :, :) = gaussian; 
kernel_2ch(1, 2, :, :) = 0; 

sobel_x = single([-1, 0, 1; 
                -2, 0, 2; 
                -1, 0, 1]);
kernel_2ch(2, 1, :, :) = 0; 
kernel_2ch(2, 2, :, :) = sobel_x;  

params_2ch = clib.xfeat.Conv2DParams(3, 3, 2, 2, 1, 1, 1, 1);

dual_filter = clib.xfeat.Conv2D(input_prop, params_2ch);
kernel_clib = flatten(kernel_2ch);
dual_filter.set_kernel(kernel_clib);

input_2ch_device = host2device(input_2ch);
output_device = dual_filter.forward(input_2ch_device);
output_2ch = device2host(output_device);

sobel_output = squeeze(output_2ch(2, :, :));

sobel_x_kernel = [-1 0 1; -2 0 2; -1 0 1];
matlab_sobel_x = single(imfilter(input_image2, sobel_x_kernel, 0, "same", 'corr'));

error_mat = abs(sobel_output - matlab_sobel_x);
error_sum = sum(abs(error_mat(:)));
error_max = max(abs(error_mat(:)));
disp(error_sum);
disp(error_max);

% sobel plot
close all
imshow(squeeze(sobel_output))
figure
imshow(matlab_sobel_x)
