#!/usr/bin/env python3

import torch
import cv2
import numpy as np
import os
import pdb


def load_cpp_tensor(name):
    """Load C++ saved tensor"""
    base = "../build/"
    data = np.fromfile(f"{base}{name}_output.bin", dtype=np.float32)
    with open(f"{base}{name}_shape.txt", 'r') as f:
        shape = list(map(int, f.read().strip().split()))
    return data.reshape(shape)


def compare_tensors(cpp_data, py_data, name):
    """Compare tensors and print stats"""
    py_data = py_data.cpu().numpy().squeeze()
    diff = np.abs(cpp_data - py_data)

    print(f"\n{name}:")
    print(f"  Shapes match: {cpp_data.shape == py_data.shape}")
    print(f"  Max diff: {diff.max():.8f}")
    print(f"  Mean diff: {diff.mean():.8f}")
    print(f"  All close (1e-2): {np.allclose(cpp_data, py_data, atol=1e-2)}")


# Load XFeat and run inference
xfeat = torch.hub.load('verlab/accelerated_features', 'XFeat', pretrained=True, top_k=1000)
xfeat.eval().cuda()

# Load same image as C++
img = cv2.imread("../data/TajMahal.png", cv2.IMREAD_GRAYSCALE)

img_float = img.astype(np.float32) / 255.0
img_tensor = torch.from_numpy(img_float).unsqueeze(0).unsqueeze(0).cuda()

print(f"Input shape: {img_tensor.shape}")

# Run through keypoint head manually
with torch.no_grad():
    # Normalize like XFeat does
    x = img_tensor.mean(dim=1, keepdim=True)
    x = xfeat.net.norm(x)
    
    # Compare input
    cpp_input = load_cpp_tensor("cpp_input")
    compare_tensors(cpp_input, x, "Input")

    # Layer 0: Fold operation (_unfold2d)
    x = xfeat.net._unfold2d(x, ws=8)
    cpp_layer0 = load_cpp_tensor("cpp_layer_0")
    compare_tensors(cpp_layer0, x, "Layer 0 (Fold)")

    # Layers 1-3: Conv + BatchNorm + ReLU
    for i in range(3):
        layer = xfeat.net.keypoint_head[i]
        x = layer(x)

        cpp_data = load_cpp_tensor(f"cpp_layer_{i+1}")
        compare_tensors(cpp_data, x, f"Layer {i+1} (Conv+BN+ReLU)")

    # Layer 4: Final conv
    final_layer = xfeat.net.keypoint_head[3]
    x = final_layer(x)

    cpp_data = load_cpp_tensor("cpp_layer_4")
    compare_tensors(cpp_data, x, "Layer 4 (Final Conv)")

    # Layer 5: Unfold (back to original resolution)
    # This would be your unfold operation - you'll need to implement the reverse
    # For now, x should be [1, 65, H/8, W/8]
    # print(f"Before unfold: {x.shape}")
    
    # x = xfeat.net.fold2d(x, ws=8)

    # cpp_data = load_cpp_tensor("cpp_layer_5")
    
    # compare_tensors(cpp_data, x, "Layer 5 (Unfold)")

print("\nComparison complete!")
