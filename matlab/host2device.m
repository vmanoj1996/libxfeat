// Copyright 2025 Manoj Velmurugan
// SPDX-License-Identifier: MIT

function device_pointer = host2device(host_matrix)

input_clib = flatten(host_matrix);
device_pointer = clib.xfeat.DevicePointer_float_(input_clib, size(host_matrix));

end
