// Copyright 2025 Manoj Velmurugan
// SPDX-License-Identifier: MIT

clear mex

% Tiny test - 1 channel, 4x4 image, 1x1 kernel
ci = 1; co = 1;
input = single(ones(ci, 4, 4));
kernel = single(ones(co, ci, 1, 1));

fprintf('Input: %dx%dx%d\n', size(input));
fprintf('Kernel: %dx%dx%dx%d\n', size(kernel));

output = conv2d_mex(input, kernel, 1, 1, 1, 1, 0, 0);

fprintf('Success! Output size: %dx%dx%d\n', size(output));
fprintf('Output values: min=%.3f, max=%.3f\n', min(output(:)), max(output(:)));
