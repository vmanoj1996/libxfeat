This folder contains MATLAB bindings to the libxfeat


Run the followimg lines, to disable the inbuilt glibc for mex sake or just be a Thanos and delete the inbuilt glibc shipped my MW. 

I dont think you need to point to cudart. 

echo 'export LD_PRELOAD="/lib/x86_64-linux-gnu/libstdc++.so.6:$LD_PRELOAD"' >> ~/.bashrc

echo 'export LD_PRELOAD="/usr/local/cuda/lib64/libcudart.so:$LD_PRELOAD"' >> ~/.bashrc


/usr/local/MATLAB/R2024b/sys/os/glnxa64

sudo mv libstdc++.so.6 libstdc++.so.6.old


% Additionally MATLAB cannot save path in linux properly by default. put
% the following line in startup.m
% edit(fullfile(userpath,'startup.m'))

addpath(genpath('/home/$USER/.../libxfeat/matlab/'));

