This folder contains MATLAB bindings to the libxfeat


% run the followimg lines, to disable the inbuilt glibc for mex sake
echo 'export LD_PRELOAD="/lib/x86_64-linux-gnu/libstdc++.so.6:$LD_PRELOAD"' >> ~/.bashrc
echo 'export LD_PRELOAD="/usr/local/cuda/lib64/libcudart.so:$LD_PRELOAD"' >> ~/.bashrc

echo 'export LD_PRELOAD="/usr/local/cuda/lib64/libcudart.so:$LD_PRELOAD"' >> ~/.bashrc

% Additionally MATLAB cannot save path in linux properly by default. put
% the following line in startup.m
% edit(fullfile(userpath,'startup.m'))

addpath(genpath('/home/$USER/.../libxfeat/matlab/generated_interfaces/'));

