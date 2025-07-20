// this wrapper was generated with Claude
#include "mex.h"
#include "conv2d.h"
#include <cuda_runtime.h>
#include <iostream>

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    // Check number of inputs and outputs
    if (nrhs != 8) {
        mexErrMsgTxt("Usage: output = conv2d_mex(input, kernel, k1, k2, s1, s2, p1, p2)");
    }
    
    if (nlhs != 1) {
        mexErrMsgTxt("One output required");
    }
    
    // Validate input types
    if (!mxIsSingle(prhs[0]) || !mxIsSingle(prhs[1])) {
        mexErrMsgTxt("Input and kernel must be single precision (float)");
    }
    
    // Get input dimensions: [ci, height, width]
    const mwSize* input_dims = mxGetDimensions(prhs[0]);
    int ndims_input = mxGetNumberOfDimensions(prhs[0]);
    
    if (ndims_input != 3) {
        mexErrMsgTxt("Input must be 3D: [ci, height, width]");
    }
    
    int ci = input_dims[0];
    int input_h = input_dims[1];
    int input_w = input_dims[2];
    
    // Get kernel dimensions: [co, ci, k1, k2]
    const mwSize* kernel_dims = mxGetDimensions(prhs[1]);
    int ndims_kernel = mxGetNumberOfDimensions(prhs[1]);
    
    if (ndims_kernel != 4) {
        mexErrMsgTxt("Kernel must be 4D: [co, ci, k1, k2]");
    }
    
    int co = kernel_dims[0];
    int ci_kernel = kernel_dims[1];
    int k1_kernel = kernel_dims[2];
    int k2_kernel = kernel_dims[3];
    
    // Get convolution parameters
    int k1 = (int)mxGetScalar(prhs[2]);
    int k2 = (int)mxGetScalar(prhs[3]);
    int s1 = (int)mxGetScalar(prhs[4]);
    int s2 = (int)mxGetScalar(prhs[5]);
    int p1 = (int)mxGetScalar(prhs[6]);
    int p2 = (int)mxGetScalar(prhs[7]);
    
    // Validate consistency
    if (ci != ci_kernel) {
        mexErrMsgTxt("Input channels mismatch between input and kernel");
    }
    
    if (k1 != k1_kernel || k2 != k2_kernel) {
        mexErrMsgTxt("Kernel size mismatch between parameters and kernel array");
    }
    
    // Set up structures
    ImgProperty input_prop = {input_h, input_w};
    Conv2DParams params = {k1, k2, ci, co, s1, s2, p1, p2};
    
    try {
        // Create convolution object
        Convolve2D conv(input_prop, params);
        
        // Set kernel weights
        float* kernel_data = (float*)mxGetData(prhs[1]);
        size_t kernel_size = co * ci * k1 * k2;
        std::vector<float> kernel_vec(kernel_data, kernel_data + kernel_size);
        conv.set_kernel(kernel_vec);
        
        // Allocate GPU memory for input
        float* input_data = (float*)mxGetData(prhs[0]);
        float* input_device;
        size_t input_size = ci * input_h * input_w * sizeof(float);
        cudaMalloc(&input_device, input_size);
        cudaMemcpy(input_device, input_data, input_size, cudaMemcpyHostToDevice);
        
        // Run forward pass
        conv.forward(input_device);
        
        // Get output dimensions
        Conv2DParams result_params = conv.get_param();
        ImgProperty output_spec = conv.get_output_spec();
        int output_h = output_spec.height;
        int output_w = output_spec.width;
                
        // Create output array: [co, output_h, output_w]
        mwSize output_dims[3] = {co, output_h, output_w};
        plhs[0] = mxCreateNumericArray(3, output_dims, mxSINGLE_CLASS, mxREAL);
        
        // Copy result back to MATLAB
        float* output_data = (float*)mxGetData(plhs[0]);
        size_t output_size = co * output_h * output_w * sizeof(float);
        cudaMemcpy(output_data, conv.get_output(), output_size, cudaMemcpyDeviceToHost);
        
        // Cleanup GPU memory
        cudaFree(input_device);
        
    } catch (const std::exception& e) {
        mexErrMsgTxt(e.what());
    }
    
    // Check for CUDA errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::string error_msg = "CUDA error: " + std::string(cudaGetErrorString(err));
        mexErrMsgTxt(error_msg.c_str());
    }
}