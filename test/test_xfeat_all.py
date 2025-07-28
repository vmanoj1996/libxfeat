#!/usr/bin/env python3
# Copyright 2025 Manoj Velmurugan
# SPDX-License-Identifier: MIT

import torch
import cv2
import numpy as np
import os
import sys

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
   print(f"  CPP shape: {cpp_data.shape}, Python shape: {py_data.shape}")
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

with torch.no_grad():
   # Normalize like XFeat does
   x = img_tensor.mean(dim=1, keepdim=True)
   x = xfeat.net.norm(x)
   
   # Compare normalized input
   cpp_input = load_cpp_tensor("normalized_input")
   compare_tensors(cpp_input, x, "Normalized Input")
   
   # Run backbone
   x1 = xfeat.net.block1(x)
   cpp_x2 = load_cpp_tensor("x2_backbone_output")
   compare_tensors(cpp_x2, x1, "X2 Backbone Output")
   
   x2 = xfeat.net.block2(x1 + xfeat.net.skip1(x))
   
   x3 = xfeat.net.block3(x2)
   cpp_x3 = load_cpp_tensor("x3_backbone_output")
   compare_tensors(cpp_x3, x3, "X3 Backbone Output")
   
   x4 = xfeat.net.block4(x3)
   cpp_x4 = load_cpp_tensor("x4_backbone_output")
   compare_tensors(cpp_x4, x4, "X4 Backbone Output")
   
   x5 = xfeat.net.block5(x4)
   cpp_x5 = load_cpp_tensor("x5_backbone_output")
   compare_tensors(cpp_x5, x5, "X5 Backbone Output")
   
   # Interpolation
   import torch.nn.functional as F
   x4_interp = F.interpolate(x4, (x3.shape[-2], x3.shape[-1]), mode='bilinear')
   cpp_x4_interp = load_cpp_tensor("x4_interpolated")
   compare_tensors(cpp_x4_interp, x4_interp, "X4 Interpolated")
   
   x5_interp = F.interpolate(x5, (x3.shape[-2], x3.shape[-1]), mode='bilinear')
   cpp_x5_interp = load_cpp_tensor("x5_interpolated")
   compare_tensors(cpp_x5_interp, x5_interp, "X5 Interpolated")
   
   # Pyramid fusion
   pyramid_sum = x3 + x4_interp + x5_interp
   cpp_pyramid = load_cpp_tensor("pyramid_fusion")
   compare_tensors(cpp_pyramid, pyramid_sum, "Pyramid Fusion")
   
   # Features
   feats = xfeat.net.block_fusion(pyramid_sum)
   cpp_feats = load_cpp_tensor("final_features")
   compare_tensors(cpp_feats, feats, "Final Features")
   
   # Heatmap
   heatmap = xfeat.net.heatmap_head(feats)
   cpp_heatmap = load_cpp_tensor("final_heatmap")
   compare_tensors(cpp_heatmap, heatmap, "Final Heatmap")
   
   # Keypoints
   keypoints = xfeat.net.keypoint_head(xfeat.net._unfold2d(x, ws=8))
   cpp_keypoints = load_cpp_tensor("final_keypoints")
   compare_tensors(cpp_keypoints, keypoints, "Final Keypoints")

print("\nComparison complete!")