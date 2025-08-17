indices = 1:20;
input = reshape(indices, [5, 4])'

out = im2col(input, [3, 3])