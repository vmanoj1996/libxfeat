import torch
import numpy as np
import os

# Load XFeat model
xfeat = torch.hub.load('verlab/accelerated_features', 'XFeat', pretrained=True, top_k=1000)

# Extract all parameters and buffers
params = {}

# Get all learnable parameters
for name, param in xfeat.named_parameters():
   params[name] = param.detach().cpu().numpy()
   print(f"Parameter: {name}, Shape: {param.shape}")

# Get all buffers (running stats, etc.)
for name, buffer in xfeat.named_buffers():
   params[name] = buffer.detach().cpu().numpy()
   print(f"Buffer: {name}, Shape: {buffer.shape}")

# Save as compressed numpy format
output_file = 'xfeat_weights.npz'
np.savez_compressed(output_file, **params)

print(f"\nSaved {len(params)} tensors to {output_file}")
print(f"File size: {os.path.getsize(output_file) / 1024 / 1024:.2f} MB")

# Verify the save worked
loaded = np.load(output_file)
print(f"Verification: Loaded {len(loaded.files)} tensors")
for name in loaded.files[:5]:  # Show first 5
   print(f"  {name}: {loaded[name].shape}")