def run_matx_benchmark():
    """Run pre-built MatX benchmark"""
    
    print("🚀 Running MatX benchmark...")
    
    try:
        # Assume benchmark is already built in test/build/
        result = subprocess.run(['./test/build/benchmark_matx'], 
                              capture_output=True, text=True, check=True)
        print(result.stdout)
        return result.stdout
        
    except subprocess.CalledProcessError as e:
        print(f"❌ MatX benchmark execution failed: {e}")
        print(f"stderr: {e.stderr}")
        print("💡 Make sure to build the MatX benchmark first:")
        print("   cd test && mkdir build && cd build")
        print("   cmake .. && make")
        return None

def main():
    """Main benchmark function"""
    
    print("GPU Performance Benchmark: MatX vs PyTorch")
    print("=" * 50)
    
    # PyTorch benchmark
    torch_results = benchmark_pytorch()
    
    if torch_results is None:
        return
    
    # MatX benchmark (pre-built)
    print("\n" + "="*30 + " MatX " + "="*33)
    matx_output = run_matx_benchmark()
    
    # Print comparison
    print_comparison_summary(torch_results)