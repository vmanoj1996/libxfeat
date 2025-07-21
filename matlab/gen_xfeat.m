clear classes
clear functions  
clear mex
rehash toolboxcache

unload(clibConfiguration('xfeat'))

outputDir = "./generated_interfaces";

% currentDir = pwd;
% hFile = [fullfile(currentDir, "../include/conv2d.hpp"), ...
%          fullfile(currentDir, "../include/primitives.hpp")];
% libFile = fullfile(currentDir, "../build/libconv2d_lib.a");

hFile = ["/home/manoj/Dropbox/work/learn/libxfeat/include/conv2d.hpp", ...
         "/home/manoj/Dropbox/work/learn/libxfeat/include/primitives.hpp"];
libFile = "/home/manoj/Dropbox/work/learn/libxfeat/build/libconv2d_lib.a";

clibgen.generateLibraryDefinition(hFile, Libraries=libFile, ...
    InterfaceName="xfeat", ...
    OutputFolder=outputDir, ...
    AdditionalLinkerFlags=["-lcudart", "-L/usr/local/cuda/lib64", "-Wl,-rpath,/usr/local/cuda/lib64"], ...
    OverwriteExistingDefinitionFiles=true)

% Add to path
addpath(genpath(outputDir));

% Build the interface
build(definexfeat)
summary(definexfeat);


% clibConfiguration('xfeat','ExecutionMode','outofprocess')