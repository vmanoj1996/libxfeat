# test_interp.py
import numpy as np
import cv2
import os
import sys
# import torch and functional
import torch
import torch.nn.functional as F

# ... (helper functions load_cpp_tensor, chw_to_hwc, hwc_to_chw are the same) ...

def load_cpp_tensor(filename):
    return np.fromfile(filename, dtype=np.float32)

def chw_to_hwc(img_chw):
    return np.transpose(img_chw, (1, 2, 0))

def hwc_to_chw(img_hwc):
    return np.transpose(img_hwc, (2, 0, 1))

def compare_outputs():
    # ... (loading dims and image is the same) ...
    with open("./test/interp/dims.txt", "r") as f:
        dims = list(map(int, f.read().split()))
    c, h, w, target_h, target_w = dims
    
    image_path = "../data/TajMahal.png"
    img_bgr = cv2.imread(image_path)
    py_input_hwc = (cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)).astype(np.float32) / 255.0
    py_input_chw = hwc_to_chw(py_input_hwc)

    cpp_output = load_cpp_tensor("./test/interp/output.bin").reshape(c, target_h, target_w)
    
    # --- MODIFICATION: Use PyTorch for reference interpolation ---
    # Convert numpy array to torch tensor
    input_tensor = torch.from_numpy(py_input_chw).unsqueeze(0) # Add batch dimension

    # Perform interpolation
    py_output_tensor = F.interpolate(input_tensor, 
                                     size=(target_h, target_w), 
                                     mode='bilinear', 
                                     align_corners=False)
    
    # Convert back to numpy array for comparison
    py_output = py_output_tensor.squeeze(0).numpy() # Remove batch dimension
    # --- END MODIFICATION ---
    
    output_diff = np.max(np.abs(py_output - cpp_output))
    rel_error = np.mean(np.abs(py_output - cpp_output) / (np.abs(py_output) + 1e-8))
    
    print(f"\nInterpolation test on TajMahal.png: {h}x{w} -> {target_h}x{target_w}")
    print(f"Reference: PyTorch F.interpolate(align_corners=False)")
    print(f"Max absolute difference: {output_diff:.2e}")
    print(f"Mean relative error: {rel_error:.2e}")
    
    # ... (the rest of the script for stats, saving images, and checking pass/fail is the same) ...
    print(f"\nStatistics:")
    print(f"Input shape: {py_input_chw.shape}, range: [{py_input_chw.min():.4f}, {py_input_chw.max():.4f}]")
    print(f"C++ output range: [{cpp_output.min():.4f}, {cpp_output.max():.4f}]")
    print(f"PyTorch output range: [{py_output.min():.4f}, {py_output.max():.4f}]")
    
    cpp_hwc = chw_to_hwc(cpp_output)
    py_hwc = chw_to_hwc(py_output)
    
    diff_hwc = np.abs(cpp_hwc - py_hwc)
    if output_diff > 1e-9:
        diff_enhanced = (diff_hwc / output_diff * 255)
    else:
        diff_enhanced = np.zeros_like(diff_hwc)
    diff_img_rgb = np.clip(diff_enhanced, 0, 255).astype(np.uint8)
    
    cv2.imwrite("./test/interp/cpp_output.png", (cpp_hwc * 255).astype(np.uint8)[:,:,::-1])
    cv2.imwrite("./test/interp/pytorch_output.png", (py_hwc * 255).astype(np.uint8)[:,:,::-1])
    cv2.imwrite("./test/interp/difference.png", diff_img_rgb[:,:,::-1])
    
    comparison = np.hstack([(cpp_hwc * 255).astype(np.uint8), (py_hwc * 255).astype(np.uint8)])
    cv2.imwrite("./test/interp/comparison.png", comparison[:,:,::-1])
    print("\nSaved outputs to ./test/interp/")
    
    if output_diff > 1e-5: # Loosened slightly for CPU vs GPU float differences
        print(f"\nTest FAILED: difference {output_diff:.2e} exceeds threshold 1e-5")
        sys.exit(1)
    else:
        print(f"\nTest PASSED")

if __name__ == "__main__":
    if not os.path.exists("./test/interp/output.bin"):
        print("Run the C++ program first!")
        sys.exit(1)
    else:
        compare_outputs()