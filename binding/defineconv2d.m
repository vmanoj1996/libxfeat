%% About defineconv2d.m
% This file defines the MATLAB interface to the library |conv2d|.
%
% Commented sections represent C++ functionality that MATLAB cannot automatically define. To include
% functionality, uncomment a section and provide values for <SHAPE>, <DIRECTION>, etc. For more
% information, see helpview(fullfile(docroot,'matlab','helptargets.map'),'cpp_define_interface') to "Define MATLAB Interface for C++ Library".



%% Setup
% Do not edit this setup section.
function libDef = defineconv2d()
libDef = clibgen.LibraryDefinition("conv2dData.xml");

%% OutputFolder and Libraries 
libDef.OutputFolder = "/home/manoj/Dropbox/work/learn/libxfeat/binding";
libDef.Libraries = "../build/libconv2d_lib.a";

%% C++ class |Conv2DParams| with MATLAB name |clib.conv2d.Conv2DParams| 
Conv2DParamsDefinition = addClass(libDef, "Conv2DParams", "MATLABName", "clib.conv2d.Conv2DParams", ...
    "Description", "clib.conv2d.Conv2DParams    Representation of C++ class Conv2DParams."); % Modify help description values as needed.

%% C++ class constructor for C++ class |Conv2DParams| 
% C++ Signature: Conv2DParams::Conv2DParams(Conv2DParams const & input1)

Conv2DParamsConstructor1Definition = addConstructor(Conv2DParamsDefinition, ...
    "Conv2DParams::Conv2DParams(Conv2DParams const & input1)", ...
    "Description", "clib.conv2d.Conv2DParams Constructor of C++ class Conv2DParams."); % Modify help description values as needed.
defineArgument(Conv2DParamsConstructor1Definition, "input1", "clib.conv2d.Conv2DParams", "input");
validate(Conv2DParamsConstructor1Definition);

%% C++ class constructor for C++ class |Conv2DParams| 
% C++ Signature: Conv2DParams::Conv2DParams()

Conv2DParamsConstructor2Definition = addConstructor(Conv2DParamsDefinition, ...
    "Conv2DParams::Conv2DParams()", ...
    "Description", "clib.conv2d.Conv2DParams Constructor of C++ class Conv2DParams."); % Modify help description values as needed.
validate(Conv2DParamsConstructor2Definition);

%% C++ class |ImgProperty| with MATLAB name |clib.conv2d.ImgProperty| 
ImgPropertyDefinition = addClass(libDef, "ImgProperty", "MATLABName", "clib.conv2d.ImgProperty", ...
    "Description", "clib.conv2d.ImgProperty    Representation of C++ class ImgProperty."); % Modify help description values as needed.

%% C++ class constructor for C++ class |ImgProperty| 
% C++ Signature: ImgProperty::ImgProperty(ImgProperty const & input1)

ImgPropertyConstructor1Definition = addConstructor(ImgPropertyDefinition, ...
    "ImgProperty::ImgProperty(ImgProperty const & input1)", ...
    "Description", "clib.conv2d.ImgProperty Constructor of C++ class ImgProperty."); % Modify help description values as needed.
defineArgument(ImgPropertyConstructor1Definition, "input1", "clib.conv2d.ImgProperty", "input");
validate(ImgPropertyConstructor1Definition);

%% C++ class constructor for C++ class |ImgProperty| 
% C++ Signature: ImgProperty::ImgProperty()

ImgPropertyConstructor2Definition = addConstructor(ImgPropertyDefinition, ...
    "ImgProperty::ImgProperty()", ...
    "Description", "clib.conv2d.ImgProperty Constructor of C++ class ImgProperty."); % Modify help description values as needed.
validate(ImgPropertyConstructor2Definition);

%% C++ class |Convolve2D| with MATLAB name |clib.conv2d.Convolve2D| 
Convolve2DDefinition = addClass(libDef, "Convolve2D", "MATLABName", "clib.conv2d.Convolve2D", ...
    "Description", "clib.conv2d.Convolve2D    Representation of C++ class Convolve2D."); % Modify help description values as needed.

%% C++ class constructor for C++ class |Convolve2D| 
% C++ Signature: Convolve2D::Convolve2D(ImgProperty input_prop_,Conv2DParams params_)

Convolve2DConstructor1Definition = addConstructor(Convolve2DDefinition, ...
    "Convolve2D::Convolve2D(ImgProperty input_prop_,Conv2DParams params_)", ...
    "Description", "clib.conv2d.Convolve2D Constructor of C++ class Convolve2D."); % Modify help description values as needed.
defineArgument(Convolve2DConstructor1Definition, "input_prop_", "clib.conv2d.ImgProperty");
defineArgument(Convolve2DConstructor1Definition, "params_", "clib.conv2d.Conv2DParams");
validate(Convolve2DConstructor1Definition);

%% C++ class method |set_kernel| for C++ class |Convolve2D| 
% C++ Signature: void Convolve2D::set_kernel(std::vector<float, std::allocator<float>> const & kernel_data)

set_kernelDefinition = addMethod(Convolve2DDefinition, ...
    "void Convolve2D::set_kernel(std::vector<float, std::allocator<float>> const & kernel_data)", ...
    "MATLABName", "set_kernel", ...
    "Description", "set_kernel Method of C++ class Convolve2D."); % Modify help description values as needed.
defineArgument(set_kernelDefinition, "kernel_data", "clib.array.conv2d.Float");
validate(set_kernelDefinition);

%% C++ class method |forward| for C++ class |Convolve2D| 
% C++ Signature: void Convolve2D::forward(FLOAT const * input_device)

%forwardDefinition = addMethod(Convolve2DDefinition, ...
%    "void Convolve2D::forward(FLOAT const * input_device)", ...
%    "MATLABName", "forward", ...
%    "Description", "forward Method of C++ class Convolve2D."); % Modify help description values as needed.
%defineArgument(forwardDefinition, "input_device", "clib.array.conv2d.Float", "input", <SHAPE>); % <MLTYPE> can be "clib.array.conv2d.Float", or "single"
%validate(forwardDefinition);

%% C++ class method |get_output| for C++ class |Convolve2D| 
% C++ Signature: FLOAT * Convolve2D::get_output()

%get_outputDefinition = addMethod(Convolve2DDefinition, ...
%    "FLOAT * Convolve2D::get_output()", ...
%    "MATLABName", "get_output", ...
%    "Description", "get_output Method of C++ class Convolve2D."); % Modify help description values as needed.
%defineOutput(get_outputDefinition, "RetVal", "single", <SHAPE>, "DeleteFcn", <DELETER>); % Specify <DELETER> or remove the "DeleteFcn" option
%validate(get_outputDefinition);

%% C++ class method |get_param| for C++ class |Convolve2D| 
% C++ Signature: Conv2DParams Convolve2D::get_param()

get_paramDefinition = addMethod(Convolve2DDefinition, ...
    "Conv2DParams Convolve2D::get_param()", ...
    "MATLABName", "get_param", ...
    "Description", "get_param Method of C++ class Convolve2D."); % Modify help description values as needed.
defineOutput(get_paramDefinition, "RetVal", "clib.conv2d.Conv2DParams");
validate(get_paramDefinition);

%% C++ class method |get_output_spec| for C++ class |Convolve2D| 
% C++ Signature: ImgProperty Convolve2D::get_output_spec()

get_output_specDefinition = addMethod(Convolve2DDefinition, ...
    "ImgProperty Convolve2D::get_output_spec()", ...
    "MATLABName", "get_output_spec", ...
    "Description", "get_output_spec Method of C++ class Convolve2D."); % Modify help description values as needed.
defineOutput(get_output_specDefinition, "RetVal", "clib.conv2d.ImgProperty");
validate(get_output_specDefinition);

%% C++ class method |get_input_spec| for C++ class |Convolve2D| 
% C++ Signature: ImgProperty Convolve2D::get_input_spec()

get_input_specDefinition = addMethod(Convolve2DDefinition, ...
    "ImgProperty Convolve2D::get_input_spec()", ...
    "MATLABName", "get_input_spec", ...
    "Description", "get_input_spec Method of C++ class Convolve2D."); % Modify help description values as needed.
defineOutput(get_input_specDefinition, "RetVal", "clib.conv2d.ImgProperty");
validate(get_input_specDefinition);

%% C++ class method |validate_params| for C++ class |Convolve2D| 
% C++ Signature: void Convolve2D::validate_params()

validate_paramsDefinition = addMethod(Convolve2DDefinition, ...
    "void Convolve2D::validate_params()", ...
    "MATLABName", "validate_params", ...
    "Description", "validate_params Method of C++ class Convolve2D."); % Modify help description values as needed.
validate(validate_paramsDefinition);

%% C++ class constructor for C++ class |Convolve2D| 
% C++ Signature: Convolve2D::Convolve2D(Convolve2D const & input1)

Convolve2DConstructor2Definition = addConstructor(Convolve2DDefinition, ...
    "Convolve2D::Convolve2D(Convolve2D const & input1)", ...
    "Description", "clib.conv2d.Convolve2D Constructor of C++ class Convolve2D."); % Modify help description values as needed.
defineArgument(Convolve2DConstructor2Definition, "input1", "clib.conv2d.Convolve2D", "input");
validate(Convolve2DConstructor2Definition);

%% Validate the library definition
validate(libDef);

end
