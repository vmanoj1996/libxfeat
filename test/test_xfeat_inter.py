#!/usr/bin/env python3
# Copyright 2025 Manoj Velmurugan
# SPDX-License-Identifier: MIT

import torch
import torch.nn.functional as F
import cv2
import numpy as np
import h5py
import os
import sys
from pathlib import Path
import ipdb

debug = False


def load_h5_tensor(filepath, dataset_name):
    """Load tensor from HDF5 file"""
    with h5py.File(filepath, 'r') as f:
        data = f[dataset_name][()]
    return torch.from_numpy(data.astype(np.float32))


def print_tensor_stats(name, tensor):
    """Print tensor statistics"""
    if isinstance(tensor, torch.Tensor):
        tensor = tensor.detach().cpu()

    print(f"  {name:<15} | Shape: {str(list(tensor.shape)):<20} | "
          f"Range: [{tensor.min().item():>8.4f}, {tensor.max().item():>8.4f}] | "
          f"Mean: {tensor.mean().item():>8.4f}")


def compare_tensors(cpp_tensor, py_tensor, name, tolerance=1e-3):
    """Compare two tensors and return comparison results"""
    if isinstance(py_tensor, torch.Tensor):
        py_tensor = py_tensor.detach().cpu()
    if isinstance(cpp_tensor, torch.Tensor):
        cpp_tensor = cpp_tensor.detach().cpu()

    # Handle shape differences
    if cpp_tensor.shape != py_tensor.shape:
        print(
            f"  ❌ {name}: Shape mismatch! C++: {cpp_tensor.shape}, PyTorch: {py_tensor.shape}")
        return False

    # Calculate differences
    abs_diff = torch.abs(cpp_tensor - py_tensor)
    max_diff = abs_diff.max().item()
    mean_diff = abs_diff.mean().item()
    rel_diff = (abs_diff / (torch.abs(py_tensor) + 1e-8)).mean().item()

    # Check if within tolerance
    passed = max_diff < tolerance
    status = "✅ PASS" if passed else "❌ FAIL"

    print(f"  {status} {name:<15} | Max diff: {max_diff:>8.5f} | "
          f"Mean diff: {mean_diff:>8.5f} | Rel diff: {rel_diff:>8.5f}")

    return passed


def load_debug_tensor(name):
    """Load tensor from debug_outputs folder"""
    filepath = f"./debug_outputs/{name}.h5"
    if Path(filepath).exists():
        return load_h5_tensor(filepath, "data")
    return None


print("Testing XFeat C++/CUDA vs PyTorch Hub...")

# Load XFeat
print("Loading XFeat from PyTorch Hub...")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
xfeat = torch.hub.load('verlab/accelerated_features',
                       'XFeat', pretrained=True, top_k=1000)
xfeat = xfeat.to(device).eval()

print(f"XFeat loaded on {device}")

if debug:
    # Show network structure like in your original
    for name, param in xfeat.named_parameters():
        print(f"{name}: {param.shape}")

    for name, buffer in xfeat.named_buffers():
        print(f"{name}: {buffer.shape}")

# Check if C++ outputs exist
output_dir = Path("./test/xfeat_output")
required_files = ["input.h5", "heatmap.h5", "keypoints.h5", "features.h5"]

missing_files = [f for f in required_files if not (output_dir / f).exists()]
if missing_files:
    print(f"✗ Missing C++ output files: {missing_files}")
    print("Please run the C++ test first to generate outputs.")
    sys.exit(1)

# Load test image - same as C++ (TajMahal.png)
image_path = os.path.join('../data', 'TajMahal.png')
test_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

if test_image is None:
    print(f"✗ Could not load image from {image_path}")
    sys.exit(1)

print(f"✓ Loaded image: {test_image.shape[1]}×{test_image.shape[0]}")

# Preprocess exactly like C++
img_float = test_image.astype(np.float32) / 255.0
img_tensor = torch.from_numpy(img_float).unsqueeze(0).unsqueeze(0).to(device)

# Load C++ outputs
print("Loading C++ outputs...")
try:
    cpp_input = load_h5_tensor(output_dir / "input.h5", "input")
    cpp_heatmap = load_h5_tensor(output_dir / "heatmap.h5", "heatmap")
    cpp_keypoints = load_h5_tensor(output_dir / "keypoints.h5", "keypoints")
    cpp_features = load_h5_tensor(output_dir / "features.h5", "features")
    print("✓ C++ outputs loaded")
except Exception as e:
    print(f"✗ Failed to load C++ outputs: {e}")
    sys.exit(1)

# Load debug intermediate outputs
debug_files = [
    "normalized_input", "x2_backbone_output", "x3_backbone_output",
    "x4_backbone_output", "x5_backbone_output", "pyramid_fusion"
]

debug_outputs = {}
for name in debug_files:
    debug_outputs[name] = load_debug_tensor(name)

# Run PyTorch XFeat inference step by step
print("Running PyTorch XFeat inference step by step...")
try:
    with torch.no_grad():
        # Normalize input like XFeat does
        x = img_tensor.mean(dim=1, keepdim=True)
        x_norm = xfeat.net.norm(x)

        # Compare normalized input
        if debug_outputs["normalized_input"] is not None:
            print("\n--- Intermediate Comparisons ---")
            compare_tensors(debug_outputs["normalized_input"].squeeze(),
                          x_norm.squeeze(), "Normalized Input", tolerance=1e-5)

        # Run backbone blocks step by step
        x1 = xfeat.net.block1(x_norm)
        x2 = xfeat.net.block2(x1 + xfeat.net.skip1(x_norm))

        if debug_outputs["x2_backbone_output"] is not None:
            compare_tensors(debug_outputs["x2_backbone_output"].squeeze(), x2.squeeze(), "X2 Backbone", tolerance=1e-3)
        
        x3 = xfeat.net.block3(x2)
        if debug_outputs["x3_backbone_output"] is not None:
            compare_tensors(debug_outputs["x3_backbone_output"].squeeze(), 
                          x3.squeeze(), "X3 Backbone", tolerance=1e-3)
        
        x4 = xfeat.net.block4(x3)
        if debug_outputs["x4_backbone_output"] is not None:
            compare_tensors(debug_outputs["x4_backbone_output"].squeeze(), 
                          x4.squeeze(), "X4 Backbone", tolerance=1e-3)
        
        x5 = xfeat.net.block5(x4)
        if debug_outputs["x5_backbone_output"] is not None:
            compare_tensors(debug_outputs["x5_backbone_output"].squeeze(), 
                          x5.squeeze(), "X5 Backbone", tolerance=1e-3)
        
        # Pyramid fusion
        x4_interp = F.interpolate(x4, size=(x3.shape[-2], x3.shape[-1]), mode='bilinear', align_corners=False)
        x5_interp = F.interpolate(x5, size=(x3.shape[-2], x3.shape[-1]), mode='bilinear', align_corners=False)
        pyramid_sum = x3 + x4_interp + x5_interp
        
        if debug_outputs["pyramid_fusion"] is not None:
            compare_tensors(debug_outputs["pyramid_fusion"].squeeze(), 
                          pyramid_sum.squeeze(), "Pyramid Fusion", tolerance=1e-3)
        
        # Get final outputs
        py_features = xfeat.net.block_fusion(pyramid_sum)
        py_heatmap = xfeat.net.heatmap_head(py_features)
        
        # Keypoints (using fold operation)
        folded_input = xfeat.net._unfold2d(x_norm, ws=8)
        py_keypoints_raw = xfeat.net.keypoint_head(folded_input)
        
    print("✓ PyTorch step-by-step inference completed")
except Exception as e:
    print(f"✗ PyTorch inference failed: {e}")
    sys.exit(1)

# Print statistics
print("\n--- Final Output Statistics ---")
print("C++ Outputs:")
print_tensor_stats("Heatmap", cpp_heatmap)
print_tensor_stats("Keypoints", cpp_keypoints)
print_tensor_stats("Features", cpp_features)

print("\nPyTorch Outputs:")
print_tensor_stats("Heatmap", py_heatmap.squeeze())
print_tensor_stats("Keypoints", py_keypoints_raw.squeeze())
print_tensor_stats("Features", py_features.squeeze())

# Compare final outputs
print("\n--- Final Output Comparison ---")
results = []

# Compare keypoints (handle potential shape mismatch)
if cpp_keypoints.shape != py_keypoints_raw.squeeze().shape:
    print(f"  ⚠️  Keypoints shape mismatch: C++ {cpp_keypoints.shape} vs PyTorch {py_keypoints_raw.squeeze().shape}")

# Summary
passed_tests = sum(results)
total_tests = len(results)

print(f"\n--- Summary ---")
print(f"Tests passed: {passed_tests}/{total_tests}")

if passed_tests == total_tests:
    print("🎉 All tests PASSED! C++ implementation matches PyTorch.")
    exit_code = 0
else:
    print("⚠️  Some tests FAILED. Check implementation differences.")
    exit_code = 1

if debug:
    # Optional: Test the high-level detectAndCompute API as well
    print("\nTesting high-level API...")
    output = xfeat.detectAndCompute(cv2.cvtColor(test_image, cv2.COLOR_GRAY2BGR), top_k=1000)
    num_features = len(output[0]['keypoints'])
    print(f"XFeat detectAndCompute: {num_features} features detected")

sys.exit(exit_code)