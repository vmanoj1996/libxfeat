import pytest
import numpy as np
from xfeat_cpp import XFeatBackbone, XFeatDetector  # Your C++ bindings

def test_backbone_basic():
    backbone = XFeatBackbone()
    assert backbone.is_initialized()

def test_feature_extraction():
    backbone = XFeatBackbone()
    image = np.random.rand(3, 480, 640).astype(np.float32)
    features = backbone.extract_features(image)
    
    assert features.shape == (64, 60, 80)  # channels, height, width
    assert features.dtype == np.float32

def test_keypoint_detection():
    detector = XFeatDetector()
    features = np.random.rand(64, 60, 80).astype(np.float32)
    keypoints = detector.detect(features, max_keypoints=1000)
    
    assert len(keypoints) <= 1000
    assert keypoints.shape[1] == 2  # x, y coordinates

def test_compare_with_pytorch():
    """Compare your C++/MATX implementation with PyTorch XFeat"""
    import torch
    xfeat_torch = torch.hub.load('verlab/accelerated_features', 'XFeat')
    
    image = np.random.rand(3, 480, 640).astype(np.float32)
    
    # Your implementation
    backbone_cpp = XFeatBackbone()
    features_cpp = backbone_cpp.extract_features(image)
    
    # PyTorch reference
    with torch.no_grad():
        features_torch = xfeat_torch.net(torch.from_numpy(image).unsqueeze(0))
    
    # Compare (allowing some numerical differences)
    np.testing.assert_allclose(features_cpp, features_torch.numpy(), rtol=1e-3)

# Run with: pytest test_xfeat.py -v