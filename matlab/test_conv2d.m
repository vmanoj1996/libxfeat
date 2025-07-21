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
