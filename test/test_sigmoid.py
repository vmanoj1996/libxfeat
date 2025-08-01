
import numpy as np
import os

def load_cpp_tensor(filename):
    return np.fromfile(filename, dtype=np.float32)

def generate_test_input(size):
    return np.arange(size, dtype=np.float32) - size/2
    return input_data * 0.1

def apply_sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def compare_outputs():
    # Test parameters
    c, h, w = 3, 4, 5
    total_size = c * h * w
    
    # Generate Python version
    py_input = generate_test_input(total_size) * 0.1
    py_output = apply_sigmoid(py_input)
    
    # Load C++ version
    cpp_input = load_cpp_tensor("./sigmoid/input.bin")
    cpp_output = load_cpp_tensor("./sigmoid/output.bin")
    
    # Compare
    input_diff = np.max(np.abs(py_input - cpp_input))
    output_diff = np.max(np.abs(py_output - cpp_output))
    
    print(f"Input difference: {input_diff:.2e}")
    print(f"Output difference: {output_diff:.2e}")
    print(f"Test {'PASSED' if output_diff < 1e-6 else 'FAILED'}")
    
    # Show some values
    print(f"\nFirst 5 values comparison:")
    print(f"C++ output: {cpp_output[:5]}")
    print(f"Py output:  {py_output[:5]}")

if __name__ == "__main__":
    if not os.path.exists("./sigmoid/output.bin"):
        print("Run the C++ program first!")
    else:
        compare_outputs()