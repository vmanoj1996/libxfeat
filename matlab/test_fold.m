clear
input  = int32(h5read('../build/fold_test/input.h5', '/input')*256);
r2c(input)

folded  = r2c(int32(h5read('../build/fold_test/folded_output.h5', '/folded')*256))

output  = r2c(int32(h5read('../build/fold_test/final_output.h5', '/output')*256))
% output = h5read('final_output.h5', '/output');
% 
% input_int  = int32(input*64)
% folded_int = int32(folded*64)
% output_int = int32(output*64)

