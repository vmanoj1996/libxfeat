# Copyright 2025 Manoj Velmurugan
# SPDX-License-Identifier: MIT

import torch
import time
import numpy as np

def benchmark_pytorch():
    device = torch.device('cuda:0')
    torch.cuda.set_device(0)
    
    sizes = [(1000, 1000), (2000, 2000), (4000, 4000), (8000, 8000)]
    
    for rows, cols in sizes:
        print(f"\n🚀 PyTorch benchmark - Size: {rows}x{cols}")
        
        # Create tensors
        a = torch.randn(rows, cols, device=device, dtype=torch.float32)
        b = torch.randn(rows, cols, device=device, dtype=torch.float32)
        c = torch.empty(rows, cols, device=device, dtype=torch.float32)
        
        torch.cuda.synchronize()
        
        # Benchmark elementwise_add
        for _ in range(10):  # Warmup
            c = a + b
        torch.cuda.synchronize()
        
        start = time.perf_counter()
        for _ in range(100):
            c = a + b
        torch.cuda.synchronize()
        end = time.perf_counter()
        
        avg_time = (end - start) * 1000 / 100
        print(f"  elementwise_add: {avg_time:.3f} ms")
        
        # Benchmark elementwise_mul
        for _ in range(10):
            c = a * b
        torch.cuda.synchronize()
        
        start = time.perf_counter()
        for _ in range(100):
            c = a * b
        torch.cuda.synchronize()
        end = time.perf_counter()
        
        avg_time = (end - start) * 1000 / 100
        print(f"  elementwise_mul: {avg_time:.3f} ms")
        
        # Benchmark matrix_mul
        for _ in range(10):
            c = torch.mm(a, b)
        torch.cuda.synchronize()
        
        start = time.perf_counter()
        for _ in range(100):
            c = torch.mm(a, b)
        torch.cuda.synchronize()
        end = time.perf_counter()
        
        avg_time = (end - start) * 1000 / 100
        print(f"  matrix_mul     : {avg_time:.3f} ms")
        
        # Benchmark sin_exp
        for _ in range(10):
            c = torch.sin(a) + torch.exp(b * 0.1)
        torch.cuda.synchronize()
        
        start = time.perf_counter()
        for _ in range(100):
            c = torch.sin(a) + torch.exp(b * 0.1)
        torch.cuda.synchronize()
        end = time.perf_counter()
        
        avg_time = (end - start) * 1000 / 100
        print(f"  sin_exp        : {avg_time:.3f} ms")
        
        # Benchmark reduction_sum
        for _ in range(10):
            sum_result = torch.sum(a + b)
        torch.cuda.synchronize()
        
        start = time.perf_counter()
        for _ in range(100):
            sum_result = torch.sum(a + b)
        torch.cuda.synchronize()
        end = time.perf_counter()
        
        avg_time = (end - start) * 1000 / 100
        print(f"  reduction_sum  : {avg_time:.3f} ms")

if __name__ == "__main__":
    print("PyTorch Performance Benchmark")
    print("=============================")
    benchmark_pytorch()