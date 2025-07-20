// Minimal MEX wrapper - validation moved to MATLAB side
#include "mex.h"
#include "conv2d.h"
#include <cuda_runtime.h>

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    // Basic argument count check only
    if (nrhs != 8 || nlhs != 1) {
        mexErrMsgTxt("Usage: output = conv2d_mex(input, kernel, k1, k2, s1, s2, p1, p2)");
    }
    
    // Get dimensions (assume MATLAB validated them)
    const mwSize* input_dims = mxGetDimensions(prhs[0]);
    const mwSize* kernel_dims = mxGetDimensions(prhs[1]);
    
    int ci = input_dims[0];
    int input_h = input_dims[1];
    int input_w = input_dims[2];
    int co = kernel_dims[0];
    
    // Get parameters
    int k1 = (int)mxGetScalar(prhs[2]);
    int k2 = (int)mxGetScalar(prhs[3]);
    int s1 = (int)mxGetScalar(prhs[4]);
    int s2 = (int)mxGetScalar(prhs[5]);
    int p1 = (int)mxGetScalar(prhs[6]);
    int p2 = (int)mxGetScalar(prhs[7]);
    
    try {
        // Create convolution
        ImgProperty input_prop = {input_h, input_w};
        Conv2DParams params = {k1, k2, ci, co, s1, s2, p1, p2};
        Convolve2D conv(input_prop, params);
        
        // Set kernel
        float* kernel_data = (float*)mxGetData(prhs[1]);
        size_t kernel_size = co * ci * k1 * k2;
        std::vector<float> kernel_vec(kernel_data, kernel_data + kernel_size);
        conv.set_kernel(kernel_vec);
        
        // GPU input
        float* input_data = (float*)mxGetData(prhs[0]);
        float* input_device;
        cudaMalloc(&input_device, ci * input_h * input_w * sizeof(float));
        cudaMemcpy(input_device, input_data, ci * input_h * input_w * sizeof(float), cudaMemcpyHostToDevice);
        
        // Forward pass
        conv.forward(input_device);
        
        // Create output
        ImgProperty output_spec = conv.get_output_spec();
        mwSize output_dims[3] = {static_cast<mwSize>(co), 
                                static_cast<mwSize>(output_spec.height), 
                                static_cast<mwSize>(output_spec.width)};
        plhs[0] = mxCreateNumericArray(3, output_dims, mxSINGLE_CLASS, mxREAL);
        
        // Copy result
        float* output_data = (float*)mxGetData(plhs[0]);
        cudaMemcpy(output_data, conv.get_output(), 
                   co * output_spec.height * output_spec.width * sizeof(float), 
                   cudaMemcpyDeviceToHost);
        
        cudaFree(input_device);
        
    } catch (const std::exception& e) {
        mexErrMsgTxt(e.what());
    }
}