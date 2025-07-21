function host_matrix = device2host(device_pointer)

% input_clib = flatten(host_matrix);
% device_pointer = clib.xfeat.DevicePointer_float_(input_clib, size(host_matrix));

shape_clib = device_pointer.get_shape();
value_clib = device_pointer.get_value();

target_shape = double(shape_clib);

host_matrix = unflatten(value_clib, target_shape);

end
