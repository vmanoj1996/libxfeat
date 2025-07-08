#!/usr/bin/env python3

import torch
import cv2
import os

debug = False

print("Testing XFeat...")

# Load XFeat
xfeat = torch.hub.load('verlab/accelerated_features', 'XFeat', pretrained=True, top_k=1000)

# Load test image from data folder
image_path = os.path.join('../data', 'ThiruvalluvarStatue.png')
test_image = cv2.imread(image_path)

if test_image is None:
    print(f"✗ Could not load image from {image_path}")
    exit(1)

# Test detection
output = xfeat.detectAndCompute(test_image, top_k=1000)
num_features = len(output[0]['keypoints'])

print(f"XFeat working! Detected {num_features} features from {image_path}")

if debug:
    # Debug: visualize keypoints
    debug_image = test_image.copy()
    keypoints = output[0]['keypoints'].cpu().numpy()
    
    # Draw keypoints as circles
    for kp in keypoints:
        x, y = int(kp[0]), int(kp[1])
        cv2.circle(debug_image, (x, y), 1, (0, 255, 0), -1)
        
    # Display if running in GUI environment
    cv2.imshow('XFeat Keypoints', debug_image)
    print("Debug: Press any key to close the window")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
