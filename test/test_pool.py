# test_pool.py
import numpy as np
import cv2
import os
import sys
import torch

def load_cpp_tensor(filename):
    return np.fromfile(filename, dtype=np.float32)

def chw_to_hwc(img_chw):
    return np.transpose(img_chw, (1, 2, 0))

def hwc_to_chw(img_hwc):
    return np.transpose(img_hwc, (2, 0, 1))

def compare_outputs():
    # Load dimensions from C++ execution
    with open("./test/pool/dims.txt", "r") as f:
        dims = list(map(int, f.read().split()))
    c, h, w, target_h, target_w, pool_factor = dims
    
    # Load original image and preprocess it identically to the C++ code
    image_path = "../data/TajMahal.png"
    img_bgr = cv2.imread(image_path)
    py_input_hwc = (cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)).astype(np.float32) / 255.0
    py_input_chw = hwc_to_chw(py_input_hwc)

    # Load C++ output data
    cpp_output = load_cpp_tensor("./test/pool/output.bin").reshape(c, target_h, target_w)
    
    # --- Reference computation using PyTorch ---
    input_tensor = torch.from_numpy(py_input_chw).unsqueeze(0) # Add batch dim
    
    # Create an AvgPool2d layer with kernel_size and stride equal to pool_factor
    pool_layer = torch.nn.AvgPool2d(kernel_size=pool_factor, stride=pool_factor)
    
    py_output_tensor = pool_layer(input_tensor)
    py_output = py_output_tensor.squeeze(0).numpy() # Remove batch dim
    # ---
    
    # Compare results
    output_diff = np.max(np.abs(py_output - cpp_output))
    rel_error = np.mean(np.abs(py_output - cpp_output) / (np.abs(py_output) + 1e-9))
    
    print(f"\nAverage Pooling Test ({pool_factor}x{pool_factor}) on TajMahal.png: {h}x{w} -> {target_h}x{target_w}")
    print(f"Max absolute difference: {output_diff:.2e}")
    print(f"Mean relative error: {rel_error:.2e}")
    
    print(f"\nStatistics:")
    print(f"C++ output range:    [{cpp_output.min():.4f}, {cpp_output.max():.4f}]")
    print(f"PyTorch output range: [{py_output.min():.4f}, {py_output.max():.4f}]")
    
    # --- Save visual outputs ---
    os.makedirs("./pool", exist_ok=True)
    cpp_hwc = chw_to_hwc(cpp_output)
    py_hwc = chw_to_hwc(py_output)
    
    # Upscale outputs back to original size for better visual comparison
    cpp_hwc_display = cv2.resize(cpp_hwc, (w, h), interpolation=cv2.INTER_NEAREST)
    py_hwc_display = cv2.resize(py_hwc, (w, h), interpolation=cv2.INTER_NEAREST)

    cv2.imwrite("./test/pool/cpp_output.png", (cpp_hwc_display * 255).astype(np.uint8)[:,:,::-1])
    cv2.imwrite("./test/pool/pytorch_output.png", (py_hwc_display * 255).astype(np.uint8)[:,:,::-1])
    
    # Difference image
    diff_hwc = np.abs(cpp_hwc - py_hwc)
    diff_hwc_display = cv2.resize(diff_hwc, (w, h), interpolation=cv2.INTER_NEAREST)
    if output_diff > 1e-9:
        diff_enhanced = (diff_hwc_display / output_diff) * 255
    else:
        diff_enhanced = np.zeros_like(diff_hwc_display)
    cv2.imwrite("./test/pool/difference.png", np.clip(diff_enhanced, 0, 255).astype(np.uint8)[:,:,::-1])
    
    print("\nSaved outputs (cpp_output.png, pytorch_output.png, difference.png) to ./pool/")
    
    # Check if test passed (threshold can be very tight for pooling)
    if output_diff > 1e-6:
        print(f"\nTest FAILED: difference {output_diff:.2e} exceeds threshold 1e-6")
        sys.exit(1)
    else:
        print(f"\nTest PASSED")

if __name__ == "__main__":
    if not os.path.exists("./test/pool/output.bin"):
        print("Run the C++ program first!")
        sys.exit(1)
    else:
        compare_outputs()