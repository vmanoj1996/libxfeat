// Copyright 2025 Manoj Velmurugan
// SPDX-License-Identifier: MIT

input = single(randn(3, 32, 32));
kernel = single(randn(16, 3, 3, 3));

output = conv2d(input, kernel, 3, 3, 2, 2, 0, 0);