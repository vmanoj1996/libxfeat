#!/usr/bin/env python3
import torch
import numpy as np
import time
import statistics

def main():
    """
    Main function to run the PyTorch XFeat performance test.
    """
    print("Starting Python performance test for XFeat forward pass...")

    # --- Configuration ---
    # Match these with your C++ test for a fair comparison
    height = 480
    width = 640
    num_runs = 50

    print("----------------------------------")
    print("Configuration:")
    print(f"  - Input Size: {width}x{height}x1")
    print(f"  - Number of Runs: {num_runs}")
    print("----------------------------------")

    # --- Model Initialization ---
    print("Loading XFeat from PyTorch Hub...")
    if not torch.cuda.is_available():
        print("✗ CUDA is not available. This script requires a GPU for a fair comparison.")
        return
        
    device = torch.device('cuda')
    try:
        xfeat = torch.hub.load('verlab/accelerated_features', 'XFeat', pretrained=True)
        xfeat = xfeat.to(device).eval()
    except Exception as e:
        print(f"✗ Failed to load model: {e}")
        return
        
    print(f"Model loaded on {device}.")

    # --- Data Preparation ---
    # Generate random input data on the GPU, just like the C++ test.
    # This avoids disk I/O and provides a consistent benchmark.
    print("Preparing random input data on device...")
    try:
        # unsqueeze(0) for batch, unsqueeze(0) for channel
        input_tensor = torch.rand(1, 1, height, width, device=device, dtype=torch.float32)
        print("Input data prepared.")
    except Exception as e:
        print(f"✗ Failed to create input tensor on GPU: {e}")
        return

    # --- Warm-up Run ---
    # A warm-up run is essential to ensure CUDA kernels are compiled,
    # and initial setup costs don't affect timing measurements.
    print("Performing warm-up run...")
    try:
        with torch.no_grad():
            _ = xfeat.net(input_tensor)
        # Synchronize to make sure the warm-up is fully complete before starting timers.
        torch.cuda.synchronize()
        print("Warm-up complete.")
    except Exception as e:
        print(f"✗ Warm-up run failed: {e}")
        return

    # --- Timed Performance Measurement ---
    print(f"Starting {num_runs} timed runs...")
    timings_ms = []
    
    try:
        for i in range(num_runs):
            # Use perf_counter for high-precision timing
            start_time = time.perf_counter()

            with torch.no_grad():
                # We time the core network pass, same as the C++ test
                _ = xfeat.net(input_tensor)

            torch.cuda.synchronize()

            end_time = time.perf_counter()
            
            elapsed_ms = (end_time - start_time) * 1000.0
            timings_ms.append(elapsed_ms)
            # print(f"  Run {i+1:2d}/{num_runs}: {elapsed_ms:8.3f} ms")
    except Exception as e:
        print(f"An error occurred during timed runs: {e}")
        return

    print("All runs complete.")

    # --- Results Analysis ---
    if not timings_ms:
        print("No timing data was collected.")
        return

    total_time_ms = sum(timings_ms)
    average_time_ms = statistics.mean(timings_ms)
    median_time_ms = statistics.median(timings_ms)
    min_time_ms = min(timings_ms)
    max_time_ms = max(timings_ms)
    
    # FPS = 1 second / average_time_in_seconds = 1000 ms / average_time_in_ms
    average_fps = 1000.0 / average_time_ms

    print("\n--- Performance Summary ---")
    print(f"Total time for {num_runs} runs: {total_time_ms:.3f} ms")
    print(f"Average latency:        {average_time_ms:.3f} ms")
    print(f"Average throughput (FPS): {average_fps:.3f}")
    print(f"Median latency:         {median_time_ms:.3f} ms")
    print(f"Minimum latency:        {min_time_ms:.3f} ms")
    print(f"Maximum latency:        {max_time_ms:.3f} ms")
    print("---------------------------\n")


if __name__ == '__main__':
    main()