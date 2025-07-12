import torch
import numpy as np
import h5py
import os

# Load XFeat model
xfeat = torch.hub.load('verlab/accelerated_features', 'XFeat', pretrained=True, top_k=1000)

# Save as HDF5 format
output_file = 'xfeat_weights.h5'

def save_dataset(group, name, data):
    """Save dataset with compression only for non-scalar data"""
    if data.ndim == 0:  # Scalar
        group.create_dataset(name, data=data)
    else:
        group.create_dataset(name, data=data, compression='gzip')

with h5py.File(output_file, 'w') as f:
    # Get all learnable parameters
    param_group = f.create_group('parameters')
    for name, param in xfeat.named_parameters():
        data = param.detach().cpu().numpy().astype(np.float32)
        save_dataset(param_group, name, data)
        print(f"Parameter: {name}, Shape: {param.shape}")

    # Get all buffers (running stats, etc.)
    buffer_group = f.create_group('buffers')
    for name, buffer in xfeat.named_buffers():
        data = buffer.detach().cpu().numpy().astype(np.float32)
        save_dataset(buffer_group, name, data)
        print(f"Buffer: {name}, Shape: {buffer.shape}")

print(f"\nSaved to {output_file}")
print(f"File size: {os.path.getsize(output_file) / 1024 / 1024:.2f} MB")

# Verify the save worked
with h5py.File(output_file, 'r') as f:
    print(f"Verification: Found groups: {list(f.keys())}")
    print(f"Parameters: {len(f['parameters'].keys())}")
    print(f"Buffers: {len(f['buffers'].keys())}")
    
    # Show first 5 parameters
    param_names = list(f['parameters'].keys())
    for name in param_names:
        shape = f['parameters'][name].shape
        print(f"  {name}: {shape}")