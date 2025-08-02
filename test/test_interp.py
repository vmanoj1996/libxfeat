# test_interp.py
import numpy as np
import cv2
import os
import sys
from scipy.ndimage import zoom

def load_cpp_tensor(filename):
    return np.fromfile(filename, dtype=np.float32)

def chw_to_hwc(img_chw):
    """Convert CHW to HWC format"""
    return np.transpose(img_chw, (1, 2, 0))

def hwc_to_chw(img_hwc):
    """Convert HWC to CHW format"""
    return np.transpose(img_hwc, (2, 0, 1))

def compare_outputs():
    # Load dimensions defined by the C++ execution
    with open("./interp/dims.txt", "r") as f:
        dims = list(map(int, f.read().split()))
    c_from_cpp, h_from_cpp, w_from_cpp, target_h, target_w = dims
    
    # Load original image instead of C++ binary input
    image_path = "../data/TajMahal.png"
    if not os.path.exists(image_path):
        print(f"Error: Original image not found at {image_path}")
        print("Please ensure the 'data' directory is at the project root relative to the build directory.")
        sys.exit(1)
        
    img_bgr = cv2.imread(image_path)
    h, w, c = img_bgr.shape
    
    # Sanity check: ensure Python and C++ are processing the same dimension image
    if (c, h, w) != (c_from_cpp, h_from_cpp, w_from_cpp):
        print("Error: Image dimensions mismatch between Python and C++.")
        print(f"Python loaded ({c}, {h}, {w}) from {image_path}")
        print(f"C++ used ({c_from_cpp}, {h_from_cpp}, {w_from_cpp}) according to dims.txt")
        sys.exit(1)
        
    # Preprocess the image similarly to the C++ code
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    py_input_hwc = img_rgb.astype(np.float32) / 255.0
    py_input = hwc_to_chw(py_input_hwc)

    # Load C++ output data
    cpp_output = load_cpp_tensor("./interp/output.bin").reshape(c, target_h, target_w)
    
    # Verify Python's input by saving it as an image
    # Note: save path is relative to where the script is run (e.g., build directory)
    os.makedirs("./interp", exist_ok=True)
    cv2.imwrite("./interp/input_check.png", (py_input_hwc * 255).astype(np.uint8)[:,:,::-1])
    print(f"Saved input visualization to ./interp/input_check.png")
    
    # Compute using scipy with bilinear interpolation
    zoom_factors = (1, target_h/h, target_w/w)
    py_output = zoom(py_input, zoom_factors, order=1, mode='nearest')
    
    # Compare
    output_diff = np.max(np.abs(py_output - cpp_output))
    rel_error = np.mean(np.abs(py_output - cpp_output) / (np.abs(py_output) + 1e-8))
    
    print(f"\nInterpolation test on TajMahal.png: {h}x{w} -> {target_h}x{target_w}")
    print(f"Max absolute difference: {output_diff:.2e}")
    print(f"Mean relative error: {rel_error:.2e}")
    
    # Show statistics
    print(f"\nStatistics:")
    print(f"Input shape: {py_input.shape}, range: [{py_input.min():.4f}, {py_input.max():.4f}]")
    print(f"C++ output range: [{cpp_output.min():.4f}, {cpp_output.max():.4f}]")
    print(f"Scipy output range: [{py_output.min():.4f}, {py_output.max():.4f}]")
    
    # Convert outputs to HWC for visualization
    cpp_hwc = chw_to_hwc(cpp_output)
    py_hwc = chw_to_hwc(py_output)
    
    # --- ADDITION: Create and save the difference image ---
    # Calculate absolute difference
    diff_hwc = np.abs(cpp_hwc - py_hwc)
    
    # Enhance contrast for visualization: scale so the max difference is white (255)
    if output_diff > 1e-9: # Avoid division by zero if images are identical
        # The enhancement factor makes the differences visible
        enhancement_factor = 255 / output_diff
        diff_enhanced = diff_hwc * enhancement_factor
    else:
        diff_enhanced = np.zeros_like(diff_hwc)
        
    # Clip values to be safe and convert to uint8
    diff_img_rgb = np.clip(diff_enhanced, 0, 255).astype(np.uint8)
    
    # Save the visual outputs
    # Convert RGB to BGR for OpenCV
    cv2.imwrite("./interp/cpp_output.png", (cpp_hwc * 255).astype(np.uint8)[:,:,::-1])
    cv2.imwrite("./interp/scipy_output.png", (py_hwc * 255).astype(np.uint8)[:,:,::-1])
    cv2.imwrite("./interp/difference.png", diff_img_rgb[:,:,::-1])
    
    # Create side-by-side comparison
    comparison = np.hstack([(cpp_hwc * 255).astype(np.uint8), (py_hwc * 255).astype(np.uint8)])
    cv2.imwrite("./interp/comparison.png", comparison[:,:,::-1])
    print("\nSaved outputs (cpp_output.png, scipy_output.png, difference.png, comparison.png) to ./interp/")
    
    # Check if test passed
    if output_diff > 1e-3:
        print(f"\nTest FAILED: difference {output_diff:.2e} exceeds threshold 1e-3")
        sys.exit(1)
    else:
        print(f"\nTest PASSED")

if __name__ == "__main__":
    if not os.path.exists("./interp/output.bin"):
        print("Run the C++ program first!")
        sys.exit(1)
    else:
        compare_outputs()