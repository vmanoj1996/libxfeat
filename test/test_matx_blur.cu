#include <matx.h>
#include <iostream>
#include <cmath>

using namespace matx;

// Save tensor as PGM image file (portable graymap format)
template<typename T>
void savePGM(const T& tensor, const std::string& filename) {
    int height = tensor.Size(0);
    int width = tensor.Size(1);
    
    FILE* fp = fopen(filename.c_str(), "wb");
    if (!fp) return;
    
    // PGM header
    fprintf(fp, "P5\n%d %d\n255\n", width, height);
    
    // Write pixel data (convert float [0,1] to uint8 [0,255])
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            float val = tensor(i, j);
            uint8_t pixel = (uint8_t)(val * 255.0f);
            fwrite(&pixel, 1, 1, fp);
        }
    }
    fclose(fp);
}

// Create a Gaussian kernel
template<typename T>
auto createGaussianKernel(int size, float sigma) {
    auto kernel = make_tensor<T>({size, size});
    
    int center = size / 2;
    float sum = 0.0f;
    
    // Generate Gaussian kernel values on host
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            float x = i - center;
            float y = j - center;
            float value = std::exp(-(x*x + y*y) / (2.0f * sigma * sigma));
            kernel(i, j) = value;
            sum += value;
        }
    }
    
    // Normalize kernel so sum = 1
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            kernel(i, j) = kernel(i, j) / sum;
        }
    }
    
    return kernel;
}

int main() {
    MATX_ENTER_HANDLER();
    
    // Initialize CUDA
    cudaExecutor exec{0};
    
    std::cout << "MatX Image Blur Test" << std::endl;
    
    // Image dimensions
    constexpr int height = 512;
    constexpr int width = 512;
    constexpr int kernel_size = 5;
    constexpr float sigma = 1.0f;
    
    // Create test image (simple pattern)
    auto input_image = make_tensor<float>({height, width});
    auto blurred_image = make_tensor<float>({height, width});
    
    std::cout << "Creating test pattern..." << std::endl;
    
    // Initialize with a test pattern (checkerboard + circle)
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            // Checkerboard pattern
            float checker = ((i/32) + (j/32)) % 2 == 0 ? 1.0f : 0.0f;
            
            // Add a circle in the center
            float cx = width / 2.0f;
            float cy = height / 2.0f;
            float dist = std::sqrt((j - cx)*(j - cx) + (i - cy)*(i - cy));
            float circle = (dist < 80.0f) ? 1.0f : 0.0f;
            
            input_image(i, j) = std::max(checker, circle);
        }
    }
    
    std::cout << "Creating Gaussian kernel (" << kernel_size << "x" << kernel_size << ", sigma=" << sigma << ")..." << std::endl;
    
    // Create Gaussian kernel
    auto gaussian_kernel = createGaussianKernel<float>(kernel_size, sigma);
    
    std::cout << "Applying Gaussian blur..." << std::endl;
    
    // Apply 2D convolution for Gaussian blur
    (blurred_image = conv2d(input_image, gaussian_kernel, MATX_C_MODE_SAME)).run(exec);
    exec.sync();
    std::cout << "✓ Gaussian blur applied successfully" << std::endl;
    
    // Save results to files in build directory
    std::cout << "Saving results..." << std::endl;
    
    // Save as PGM image files (viewable with any image viewer)
    savePGM(input_image, "test_image_original.pgm");
    savePGM(blurred_image, "test_image_blurred.pgm");
    std::cout << "✓ Saved image files: test_image_original.pgm, test_image_blurred.pgm" << std::endl;

#ifdef MATX_ENABLE_FILEIO
    // Save as NPY files (requires pybind11)
    io::write_npy(input_image, "test_image_original.npy");
    io::write_npy(blurred_image, "test_image_blurred.npy");
    io::write_npy(gaussian_kernel, "gaussian_kernel.npy");
    std::cout << "✓ Saved NPY files: test_image_original.npy, test_image_blurred.npy, gaussian_kernel.npy" << std::endl;
    
    // Also save as CSV (for easy viewing)
    io::write_csv(gaussian_kernel, "gaussian_kernel.csv", ",");
    std::cout << "✓ Saved kernel as CSV: gaussian_kernel.csv" << std::endl;
#else
    std::cout << "⚠ File I/O not enabled. Enable with -DMATX_EN_FILEIO=ON" << std::endl;
#endif
    
    // Print some statistics
    float max_original = 0.0f, max_blurred = 0.0f;
    float min_original = 1.0f, min_blurred = 1.0f;
    
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            max_original = std::max(max_original, input_image(i, j));
            min_original = std::min(min_original, input_image(i, j));
            max_blurred = std::max(max_blurred, blurred_image(i, j));
            min_blurred = std::min(min_blurred, blurred_image(i, j));
        }
    }
    
    std::cout << "\nImage Statistics:" << std::endl;
    std::cout << "Original - Min: " << min_original << ", Max: " << max_original << std::endl;
    std::cout << "Blurred  - Min: " << min_blurred << ", Max: " << max_blurred << std::endl;
    
    // Verify blur effect (blurred image should have less extreme values)
    bool blur_effective = (max_blurred < max_original) || (min_blurred > min_original);
    
    if (blur_effective) {
        std::cout << "✓ Gaussian blur test PASSED - Image was successfully blurred" << std::endl;
        return 0;
    } else {
        std::cout << "✗ Gaussian blur test FAILED - No blur effect detected" << std::endl;
        return 1;
    }
    
    MATX_EXIT_HANDLER();
}