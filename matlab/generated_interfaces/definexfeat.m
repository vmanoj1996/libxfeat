// Copyright 2025 Manoj Velmurugan
// SPDX-License-Identifier: MIT

%% About definexfeat.m
% This file defines the MATLAB interface to the library |xfeat|.
%
% Commented sections represent C++ functionality that MATLAB cannot automatically define. To include
% functionality, uncomment a section and provide values for <SHAPE>, <DIRECTION>, etc. For more
% information, see helpview(fullfile(docroot,'matlab','helptargets.map'),'cpp_define_interface') to "Define MATLAB Interface for C++ Library".



%% Setup
% Do not edit this setup section.
function libDef = definexfeat()
libDef = clibgen.LibraryDefinition("xfeatData.xml");

%% OutputFolder and Libraries 
libDef.OutputFolder = "/home/manoj/Dropbox/work/learn/libxfeat/matlab/generated_interfaces";
libDef.Libraries = "/home/manoj/Dropbox/work/learn/libxfeat/build/libconv2d_lib.a";

%% C++ class |ImgProperty| with MATLAB name |clib.xfeat.ImgProperty| 
ImgPropertyDefinition = addClass(libDef, "ImgProperty", "MATLABName", "clib.xfeat.ImgProperty", ...
    "Description", "clib.xfeat.ImgProperty    Representation of C++ class ImgProperty."); % Modify help description values as needed.

%% C++ class constructor for C++ class |ImgProperty| 
% C++ Signature: ImgProperty::ImgProperty()

ImgPropertyConstructor1Definition = addConstructor(ImgPropertyDefinition, ...
    "ImgProperty::ImgProperty()", ...
    "Description", "clib.xfeat.ImgProperty Constructor of C++ class ImgProperty."); % Modify help description values as needed.
validate(ImgPropertyConstructor1Definition);

%% C++ class constructor for C++ class |ImgProperty| 
% C++ Signature: ImgProperty::ImgProperty(int height_,int width_)

ImgPropertyConstructor2Definition = addConstructor(ImgPropertyDefinition, ...
    "ImgProperty::ImgProperty(int height_,int width_)", ...
    "Description", "clib.xfeat.ImgProperty Constructor of C++ class ImgProperty."); % Modify help description values as needed.
defineArgument(ImgPropertyConstructor2Definition, "height_", "int32");
defineArgument(ImgPropertyConstructor2Definition, "width_", "int32");
validate(ImgPropertyConstructor2Definition);

%% C++ class constructor for C++ class |ImgProperty| 
% C++ Signature: ImgProperty::ImgProperty(ImgProperty const & input1)

ImgPropertyConstructor3Definition = addConstructor(ImgPropertyDefinition, ...
    "ImgProperty::ImgProperty(ImgProperty const & input1)", ...
    "Description", "clib.xfeat.ImgProperty Constructor of C++ class ImgProperty."); % Modify help description values as needed.
defineArgument(ImgPropertyConstructor3Definition, "input1", "clib.xfeat.ImgProperty", "input");
validate(ImgPropertyConstructor3Definition);

%% C++ class public data member |height| for C++ class |ImgProperty| 
% C++ Signature: int ImgProperty::height

addProperty(ImgPropertyDefinition, "height", "int32", ...
    "Description", "int32    Data member of C++ class ImgProperty."); % Modify help description values as needed.

%% C++ class public data member |width| for C++ class |ImgProperty| 
% C++ Signature: int ImgProperty::width

addProperty(ImgPropertyDefinition, "width", "int32", ...
    "Description", "int32    Data member of C++ class ImgProperty."); % Modify help description values as needed.

%% C++ class |DevicePointer<float>| with MATLAB name |clib.xfeat.DevicePointer_float_| 
DevicePointer_float_Definition = addClass(libDef, "DevicePointer<float>", "MATLABName", "clib.xfeat.DevicePointer_float_", ...
    "Description", "clib.xfeat.DevicePointer_float_    Representation of C++ class DevicePointer<float>."); % Modify help description values as needed.

%% C++ class constructor for C++ class |DevicePointer<float>| 
% C++ Signature: DevicePointer<float>::DevicePointer()

DevicePointer_float_Constructor1Definition = addConstructor(DevicePointer_float_Definition, ...
    "DevicePointer<float>::DevicePointer()", ...
    "Description", "clib.xfeat.DevicePointer_float_ Constructor of C++ class DevicePointer<float>."); % Modify help description values as needed.
validate(DevicePointer_float_Constructor1Definition);

%% C++ class constructor for C++ class |DevicePointer<float>| 
% C++ Signature: DevicePointer<float>::DevicePointer(int total_dim)

DevicePointer_float_Constructor2Definition = addConstructor(DevicePointer_float_Definition, ...
    "DevicePointer<float>::DevicePointer(int total_dim)", ...
    "Description", "clib.xfeat.DevicePointer_float_ Constructor of C++ class DevicePointer<float>."); % Modify help description values as needed.
defineArgument(DevicePointer_float_Constructor2Definition, "total_dim", "int32");
validate(DevicePointer_float_Constructor2Definition);

%% C++ class constructor for C++ class |DevicePointer<float>| 
% C++ Signature: DevicePointer<float>::DevicePointer(std::vector<float, std::allocator<float>> const & input,std::vector<int, std::allocator<int>> dims_)

DevicePointer_float_Constructor3Definition = addConstructor(DevicePointer_float_Definition, ...
    "DevicePointer<float>::DevicePointer(std::vector<float, std::allocator<float>> const & input,std::vector<int, std::allocator<int>> dims_)", ...
    "Description", "clib.xfeat.DevicePointer_float_ Constructor of C++ class DevicePointer<float>."); % Modify help description values as needed.
defineArgument(DevicePointer_float_Constructor3Definition, "input", "clib.array.xfeat.Float");
defineArgument(DevicePointer_float_Constructor3Definition, "dims_", "clib.array.xfeat.Int");
validate(DevicePointer_float_Constructor3Definition);

%% C++ class method |get| for C++ class |DevicePointer<float>| 
% C++ Signature: float * DevicePointer<float>::get()

%getDefinition = addMethod(DevicePointer_float_Definition, ...
%    "float * DevicePointer<float>::get()", ...
%    "MATLABName", "get", ...
%    "Description", "get Method of C++ class DevicePointer<float>."); % Modify help description values as needed.
%defineOutput(getDefinition, "RetVal", "single", <SHAPE>, "DeleteFcn", <DELETER>); % Specify <DELETER> or remove the "DeleteFcn" option
%validate(getDefinition);

%% C++ class method |alloc| for C++ class |DevicePointer<float>| 
% C++ Signature: void DevicePointer<float>::alloc(std::vector<int, std::allocator<int>> dims_)

allocDefinition = addMethod(DevicePointer_float_Definition, ...
    "void DevicePointer<float>::alloc(std::vector<int, std::allocator<int>> dims_)", ...
    "MATLABName", "alloc", ...
    "Description", "alloc Method of C++ class DevicePointer<float>."); % Modify help description values as needed.
defineArgument(allocDefinition, "dims_", "clib.array.xfeat.Int");
validate(allocDefinition);

%% C++ class method |alloc| for C++ class |DevicePointer<float>| 
% C++ Signature: void DevicePointer<float>::alloc(int total_dim)

allocDefinition = addMethod(DevicePointer_float_Definition, ...
    "void DevicePointer<float>::alloc(int total_dim)", ...
    "MATLABName", "alloc", ...
    "Description", "alloc Method of C++ class DevicePointer<float>."); % Modify help description values as needed.
defineArgument(allocDefinition, "total_dim", "int32");
validate(allocDefinition);

%% C++ class method |set_value| for C++ class |DevicePointer<float>| 
% C++ Signature: void DevicePointer<float>::set_value(std::vector<float, std::allocator<float>> const & input)

set_valueDefinition = addMethod(DevicePointer_float_Definition, ...
    "void DevicePointer<float>::set_value(std::vector<float, std::allocator<float>> const & input)", ...
    "MATLABName", "set_value", ...
    "Description", "set_value Method of C++ class DevicePointer<float>."); % Modify help description values as needed.
defineArgument(set_valueDefinition, "input", "clib.array.xfeat.Float");
validate(set_valueDefinition);

%% C++ class method |get_value| for C++ class |DevicePointer<float>| 
% C++ Signature: std::vector<float, std::allocator<float>> DevicePointer<float>::get_value() const

get_valueDefinition = addMethod(DevicePointer_float_Definition, ...
    "std::vector<float, std::allocator<float>> DevicePointer<float>::get_value() const", ...
    "MATLABName", "get_value", ...
    "Description", "get_value Method of C++ class DevicePointer<float>."); % Modify help description values as needed.
defineOutput(get_valueDefinition, "RetVal", "clib.array.xfeat.Float");
validate(get_valueDefinition);

%% C++ class method |get_shape| for C++ class |DevicePointer<float>| 
% C++ Signature: std::vector<int, std::allocator<int>> DevicePointer<float>::get_shape() const

get_shapeDefinition = addMethod(DevicePointer_float_Definition, ...
    "std::vector<int, std::allocator<int>> DevicePointer<float>::get_shape() const", ...
    "MATLABName", "get_shape", ...
    "Description", "get_shape Method of C++ class DevicePointer<float>."); % Modify help description values as needed.
defineOutput(get_shapeDefinition, "RetVal", "clib.array.xfeat.Int");
validate(get_shapeDefinition);

%% C++ class |DevicePointer<int>| with MATLAB name |clib.xfeat.DevicePointer_int_| 
DevicePointer_int_Definition = addClass(libDef, "DevicePointer<int>", "MATLABName", "clib.xfeat.DevicePointer_int_", ...
    "Description", "clib.xfeat.DevicePointer_int_    Representation of C++ class DevicePointer<int>."); % Modify help description values as needed.

%% C++ class constructor for C++ class |DevicePointer<int>| 
% C++ Signature: DevicePointer<int>::DevicePointer()

DevicePointer_int_Constructor1Definition = addConstructor(DevicePointer_int_Definition, ...
    "DevicePointer<int>::DevicePointer()", ...
    "Description", "clib.xfeat.DevicePointer_int_ Constructor of C++ class DevicePointer<int>."); % Modify help description values as needed.
validate(DevicePointer_int_Constructor1Definition);

%% C++ class constructor for C++ class |DevicePointer<int>| 
% C++ Signature: DevicePointer<int>::DevicePointer(int total_dim)

DevicePointer_int_Constructor2Definition = addConstructor(DevicePointer_int_Definition, ...
    "DevicePointer<int>::DevicePointer(int total_dim)", ...
    "Description", "clib.xfeat.DevicePointer_int_ Constructor of C++ class DevicePointer<int>."); % Modify help description values as needed.
defineArgument(DevicePointer_int_Constructor2Definition, "total_dim", "int32");
validate(DevicePointer_int_Constructor2Definition);

%% C++ class constructor for C++ class |DevicePointer<int>| 
% C++ Signature: DevicePointer<int>::DevicePointer(std::vector<int, std::allocator<int>> const & input,std::vector<int, std::allocator<int>> dims_)

DevicePointer_int_Constructor3Definition = addConstructor(DevicePointer_int_Definition, ...
    "DevicePointer<int>::DevicePointer(std::vector<int, std::allocator<int>> const & input,std::vector<int, std::allocator<int>> dims_)", ...
    "Description", "clib.xfeat.DevicePointer_int_ Constructor of C++ class DevicePointer<int>."); % Modify help description values as needed.
defineArgument(DevicePointer_int_Constructor3Definition, "input", "clib.array.xfeat.Int");
defineArgument(DevicePointer_int_Constructor3Definition, "dims_", "clib.array.xfeat.Int");
validate(DevicePointer_int_Constructor3Definition);

%% C++ class method |get| for C++ class |DevicePointer<int>| 
% C++ Signature: int * DevicePointer<int>::get()

%getDefinition = addMethod(DevicePointer_int_Definition, ...
%    "int * DevicePointer<int>::get()", ...
%    "MATLABName", "get", ...
%    "Description", "get Method of C++ class DevicePointer<int>."); % Modify help description values as needed.
%defineOutput(getDefinition, "RetVal", "int32", <SHAPE>, "DeleteFcn", <DELETER>); % Specify <DELETER> or remove the "DeleteFcn" option
%validate(getDefinition);

%% C++ class method |alloc| for C++ class |DevicePointer<int>| 
% C++ Signature: void DevicePointer<int>::alloc(std::vector<int, std::allocator<int>> dims_)

allocDefinition = addMethod(DevicePointer_int_Definition, ...
    "void DevicePointer<int>::alloc(std::vector<int, std::allocator<int>> dims_)", ...
    "MATLABName", "alloc", ...
    "Description", "alloc Method of C++ class DevicePointer<int>."); % Modify help description values as needed.
defineArgument(allocDefinition, "dims_", "clib.array.xfeat.Int");
validate(allocDefinition);

%% C++ class method |alloc| for C++ class |DevicePointer<int>| 
% C++ Signature: void DevicePointer<int>::alloc(int total_dim)

allocDefinition = addMethod(DevicePointer_int_Definition, ...
    "void DevicePointer<int>::alloc(int total_dim)", ...
    "MATLABName", "alloc", ...
    "Description", "alloc Method of C++ class DevicePointer<int>."); % Modify help description values as needed.
defineArgument(allocDefinition, "total_dim", "int32");
validate(allocDefinition);

%% C++ class method |set_value| for C++ class |DevicePointer<int>| 
% C++ Signature: void DevicePointer<int>::set_value(std::vector<int, std::allocator<int>> const & input)

set_valueDefinition = addMethod(DevicePointer_int_Definition, ...
    "void DevicePointer<int>::set_value(std::vector<int, std::allocator<int>> const & input)", ...
    "MATLABName", "set_value", ...
    "Description", "set_value Method of C++ class DevicePointer<int>."); % Modify help description values as needed.
defineArgument(set_valueDefinition, "input", "clib.array.xfeat.Int");
validate(set_valueDefinition);

%% C++ class method |get_value| for C++ class |DevicePointer<int>| 
% C++ Signature: std::vector<int, std::allocator<int>> DevicePointer<int>::get_value() const

get_valueDefinition = addMethod(DevicePointer_int_Definition, ...
    "std::vector<int, std::allocator<int>> DevicePointer<int>::get_value() const", ...
    "MATLABName", "get_value", ...
    "Description", "get_value Method of C++ class DevicePointer<int>."); % Modify help description values as needed.
defineOutput(get_valueDefinition, "RetVal", "clib.array.xfeat.Int");
validate(get_valueDefinition);

%% C++ class method |get_shape| for C++ class |DevicePointer<int>| 
% C++ Signature: std::vector<int, std::allocator<int>> DevicePointer<int>::get_shape() const

get_shapeDefinition = addMethod(DevicePointer_int_Definition, ...
    "std::vector<int, std::allocator<int>> DevicePointer<int>::get_shape() const", ...
    "MATLABName", "get_shape", ...
    "Description", "get_shape Method of C++ class DevicePointer<int>."); % Modify help description values as needed.
defineOutput(get_shapeDefinition, "RetVal", "clib.array.xfeat.Int");
validate(get_shapeDefinition);

%% C++ class |Conv2DParams| with MATLAB name |clib.xfeat.Conv2DParams| 
Conv2DParamsDefinition = addClass(libDef, "Conv2DParams", "MATLABName", "clib.xfeat.Conv2DParams", ...
    "Description", "clib.xfeat.Conv2DParams    Representation of C++ class Conv2DParams."); % Modify help description values as needed.

%% C++ class constructor for C++ class |Conv2DParams| 
% C++ Signature: Conv2DParams::Conv2DParams()

Conv2DParamsConstructor1Definition = addConstructor(Conv2DParamsDefinition, ...
    "Conv2DParams::Conv2DParams()", ...
    "Description", "clib.xfeat.Conv2DParams Constructor of C++ class Conv2DParams."); % Modify help description values as needed.
validate(Conv2DParamsConstructor1Definition);

%% C++ class constructor for C++ class |Conv2DParams| 
% C++ Signature: Conv2DParams::Conv2DParams(int k1_,int k2_,int ci_,int co_,int s1_,int s2_,int p1_,int p2_)

Conv2DParamsConstructor2Definition = addConstructor(Conv2DParamsDefinition, ...
    "Conv2DParams::Conv2DParams(int k1_,int k2_,int ci_,int co_,int s1_,int s2_,int p1_,int p2_)", ...
    "Description", "clib.xfeat.Conv2DParams Constructor of C++ class Conv2DParams."); % Modify help description values as needed.
defineArgument(Conv2DParamsConstructor2Definition, "k1_", "int32");
defineArgument(Conv2DParamsConstructor2Definition, "k2_", "int32");
defineArgument(Conv2DParamsConstructor2Definition, "ci_", "int32");
defineArgument(Conv2DParamsConstructor2Definition, "co_", "int32");
defineArgument(Conv2DParamsConstructor2Definition, "s1_", "int32");
defineArgument(Conv2DParamsConstructor2Definition, "s2_", "int32");
defineArgument(Conv2DParamsConstructor2Definition, "p1_", "int32");
defineArgument(Conv2DParamsConstructor2Definition, "p2_", "int32");
validate(Conv2DParamsConstructor2Definition);

%% C++ class constructor for C++ class |Conv2DParams| 
% C++ Signature: Conv2DParams::Conv2DParams(Conv2DParams const & input1)

Conv2DParamsConstructor3Definition = addConstructor(Conv2DParamsDefinition, ...
    "Conv2DParams::Conv2DParams(Conv2DParams const & input1)", ...
    "Description", "clib.xfeat.Conv2DParams Constructor of C++ class Conv2DParams."); % Modify help description values as needed.
defineArgument(Conv2DParamsConstructor3Definition, "input1", "clib.xfeat.Conv2DParams", "input");
validate(Conv2DParamsConstructor3Definition);

%% C++ class public data member |k1| for C++ class |Conv2DParams| 
% C++ Signature: int Conv2DParams::k1

addProperty(Conv2DParamsDefinition, "k1", "int32", ...
    "Description", "int32    Data member of C++ class Conv2DParams."); % Modify help description values as needed.

%% C++ class public data member |k2| for C++ class |Conv2DParams| 
% C++ Signature: int Conv2DParams::k2

addProperty(Conv2DParamsDefinition, "k2", "int32", ...
    "Description", "int32    Data member of C++ class Conv2DParams."); % Modify help description values as needed.

%% C++ class public data member |ci| for C++ class |Conv2DParams| 
% C++ Signature: int Conv2DParams::ci

addProperty(Conv2DParamsDefinition, "ci", "int32", ...
    "Description", "int32    Data member of C++ class Conv2DParams."); % Modify help description values as needed.

%% C++ class public data member |co| for C++ class |Conv2DParams| 
% C++ Signature: int Conv2DParams::co

addProperty(Conv2DParamsDefinition, "co", "int32", ...
    "Description", "int32    Data member of C++ class Conv2DParams."); % Modify help description values as needed.

%% C++ class public data member |s1| for C++ class |Conv2DParams| 
% C++ Signature: int Conv2DParams::s1

addProperty(Conv2DParamsDefinition, "s1", "int32", ...
    "Description", "int32    Data member of C++ class Conv2DParams."); % Modify help description values as needed.

%% C++ class public data member |s2| for C++ class |Conv2DParams| 
% C++ Signature: int Conv2DParams::s2

addProperty(Conv2DParamsDefinition, "s2", "int32", ...
    "Description", "int32    Data member of C++ class Conv2DParams."); % Modify help description values as needed.

%% C++ class public data member |p1| for C++ class |Conv2DParams| 
% C++ Signature: int Conv2DParams::p1

addProperty(Conv2DParamsDefinition, "p1", "int32", ...
    "Description", "int32    Data member of C++ class Conv2DParams."); % Modify help description values as needed.

%% C++ class public data member |p2| for C++ class |Conv2DParams| 
% C++ Signature: int Conv2DParams::p2

addProperty(Conv2DParamsDefinition, "p2", "int32", ...
    "Description", "int32    Data member of C++ class Conv2DParams."); % Modify help description values as needed.

%% C++ class |Conv2D| with MATLAB name |clib.xfeat.Conv2D| 
Conv2DDefinition = addClass(libDef, "Conv2D", "MATLABName", "clib.xfeat.Conv2D", ...
    "Description", "clib.xfeat.Conv2D    Representation of C++ class Conv2D."); % Modify help description values as needed.

%% C++ class constructor for C++ class |Conv2D| 
% C++ Signature: Conv2D::Conv2D(ImgProperty input_prop_,Conv2DParams params_)

Conv2DConstructor1Definition = addConstructor(Conv2DDefinition, ...
    "Conv2D::Conv2D(ImgProperty input_prop_,Conv2DParams params_)", ...
    "Description", "clib.xfeat.Conv2D Constructor of C++ class Conv2D."); % Modify help description values as needed.
defineArgument(Conv2DConstructor1Definition, "input_prop_", "clib.xfeat.ImgProperty");
defineArgument(Conv2DConstructor1Definition, "params_", "clib.xfeat.Conv2DParams");
validate(Conv2DConstructor1Definition);

%% C++ class method |forward| for C++ class |Conv2D| 
% C++ Signature: DevicePointer<float> const & Conv2D::forward(DevicePointer<float> & input_device)

forwardDefinition = addMethod(Conv2DDefinition, ...
    "DevicePointer<float> const & Conv2D::forward(DevicePointer<float> & input_device)", ...
    "MATLABName", "forward", ...
    "Description", "forward Method of C++ class Conv2D."); % Modify help description values as needed.
defineArgument(forwardDefinition, "input_device", "clib.xfeat.DevicePointer_float_", "input");
defineOutput(forwardDefinition, "RetVal", "clib.xfeat.DevicePointer_float_");
validate(forwardDefinition);

%% C++ class method |get_output| for C++ class |Conv2D| 
% C++ Signature: DevicePointer<float> const & Conv2D::get_output()

get_outputDefinition = addMethod(Conv2DDefinition, ...
    "DevicePointer<float> const & Conv2D::get_output()", ...
    "MATLABName", "get_output", ...
    "Description", "get_output Method of C++ class Conv2D."); % Modify help description values as needed.
defineOutput(get_outputDefinition, "RetVal", "clib.xfeat.DevicePointer_float_");
validate(get_outputDefinition);

%% C++ class method |get_param| for C++ class |Conv2D| 
% C++ Signature: Conv2DParams Conv2D::get_param() const

get_paramDefinition = addMethod(Conv2DDefinition, ...
    "Conv2DParams Conv2D::get_param() const", ...
    "MATLABName", "get_param", ...
    "Description", "get_param Method of C++ class Conv2D."); % Modify help description values as needed.
defineOutput(get_paramDefinition, "RetVal", "clib.xfeat.Conv2DParams");
validate(get_paramDefinition);

%% C++ class method |set_kernel| for C++ class |Conv2D| 
% C++ Signature: void Conv2D::set_kernel(std::vector<float, std::allocator<float>> const & kernel_data)

set_kernelDefinition = addMethod(Conv2DDefinition, ...
    "void Conv2D::set_kernel(std::vector<float, std::allocator<float>> const & kernel_data)", ...
    "MATLABName", "set_kernel", ...
    "Description", "set_kernel Method of C++ class Conv2D."); % Modify help description values as needed.
defineArgument(set_kernelDefinition, "kernel_data", "clib.array.xfeat.Float");
validate(set_kernelDefinition);

%% C++ class method |get_output_spec| for C++ class |Conv2D| 
% C++ Signature: ImgProperty Conv2D::get_output_spec() const

get_output_specDefinition = addMethod(Conv2DDefinition, ...
    "ImgProperty Conv2D::get_output_spec() const", ...
    "MATLABName", "get_output_spec", ...
    "Description", "get_output_spec Method of C++ class Conv2D."); % Modify help description values as needed.
defineOutput(get_output_specDefinition, "RetVal", "clib.xfeat.ImgProperty");
validate(get_output_specDefinition);

%% C++ class method |get_input_spec| for C++ class |Conv2D| 
% C++ Signature: ImgProperty Conv2D::get_input_spec() const

get_input_specDefinition = addMethod(Conv2DDefinition, ...
    "ImgProperty Conv2D::get_input_spec() const", ...
    "MATLABName", "get_input_spec", ...
    "Description", "get_input_spec Method of C++ class Conv2D."); % Modify help description values as needed.
defineOutput(get_input_specDefinition, "RetVal", "clib.xfeat.ImgProperty");
validate(get_input_specDefinition);

%% C++ class method |validate_params| for C++ class |Conv2D| 
% C++ Signature: void Conv2D::validate_params()

validate_paramsDefinition = addMethod(Conv2DDefinition, ...
    "void Conv2D::validate_params()", ...
    "MATLABName", "validate_params", ...
    "Description", "validate_params Method of C++ class Conv2D."); % Modify help description values as needed.
validate(validate_paramsDefinition);

%% Validate the library definition
validate(libDef);

end
