hFile = ["../include/conv2d.hpp", "../include/primitives.hpp"];
libFile = "../build/libconv2d_lib.a";

% Generate the definition file
clibgen.generateLibraryDefinition(hFile, Libraries=libFile, InterfaceName="conv2d",...
    AdditionalLinkerFlags=["-lcudart", "-L/usr/local/cuda/lib64"], ...
    OverwriteExistingDefinitionFiles=true)

% Build the interface
build(defineconv2d)