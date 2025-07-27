# Copyright 2025 Manoj Velmurugan
# SPDX-License-Identifier: MIT

import torch

# Load XFeat model
print("Loading XFeat model...")
xfeat = torch.hub.load('verlab/accelerated_features', 'XFeat', pretrained=True, top_k=1000)

from torchsummary import summary
print("=== Using torchsummary ===")
summary(xfeat, input_size=(3, 640, 480))