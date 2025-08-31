#include <iostream>
#include <fstream>
#include <vector>
#include <cuda_runtime.h>
#include "conv2d_gemm.hpp"

template<int K, int CI, int CO, int S, int P>
void profile_template(int h, int w, std::ofstream& csv, std::ofstream& detailed_csv) {
    constexpr Conv2DParams params{K, K, CI, CO, S, S, P, P};
    ImgProperty input_prop{CI, h, w};
    
    // Create layer
    std::vector<FLOAT> kernel(CO * CI * K * K, 0.1f);
    auto layer_ptr = conv2d<params>(input_prop, kernel, Identity(), 0);
    auto conv = static_cast<Conv2D<params, Identity>*>(layer_ptr.get());
    
    // Create input
    std::vector<FLOAT> input(CI * h * w, 1.0f);
    DevicePointer<FLOAT> input_d(input, {CI, h, w});
    
    // Test different thread configurations
    float best_time = 1e9;
    int best_tc1 = 2, best_tc2 = 32;
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);
    
    for(int tc1 = 1; tc1 <= 128; tc1 *= 2) {
        for(int tc2 = 1; tc2 <= 128; tc2 *= 2) {
            if(tc1 * tc2 > 1024) continue;
            
            // Warmup
            try 
            {
                for(int i = 0; i < 5; i++) 
                    conv->forward_profile(input_d, tc1, tc2);
            }
            catch(const std::exception& e)
            {
                // kernel launch failed with this config.
                continue;
            }
            
            // Time
            cudaEventRecord(start);
            for(int i = 0; i < 100; i++) 
                conv->forward_profile(input_d, tc1, tc2);
            cudaEventRecord(stop);
            cudaEventSynchronize(stop);
            
            float ms;
            cudaEventElapsedTime(&ms, start, stop);
            float avg_ms = ms / 100.0f;
            
            // Write to detailed CSV
            detailed_csv << K << "," << CI << "," << CO << "," << S << "," << P << ","
                        << h << "," << w << "," << tc1 << "," << tc2 << "," 
                        << avg_ms << std::endl;

            if(ms < best_time) {
                best_time = ms;
                best_tc1 = tc1;
                best_tc2 = tc2;
            }
        }
    }
    
    float avg_time = best_time / 100.0f;
    
    // Console output
    // std::cout << K << "x" << K << " " << CI << "->" << CO 
    //           << " s=" << S << " p=" << P << " @ " << h << "x" << w
    //           << ": " << avg_time << "ms (best: " 
    //           << best_tc1 << "x" << best_tc2 << ")" << std::endl;
    
    // CSV output (summary)
    csv << K << "," << CI << "," << CO << "," << S << "," << P << ","
        << h << "," << w << "," << avg_time << "," 
        << best_tc1 << "," << best_tc2 << std::endl;

    // best time in different format
    std::cout << "{" << K << ", " << CI << ", " << CO << ", " << S << ", " << P << ", " 
                        << h << ", " << w << ", " << best_tc1 << ", " << best_tc2 << "}," << std::endl;
    
    cudaEventDestroy(start); cudaEventDestroy(stop);
}

int main() {
    // Clear and create new CSV files
    std::ofstream csv("conv2d_profile.csv", std::ios::trunc);
    csv << "kernel,in_ch,out_ch,stride,pad,height,width,time_ms,best_tc1,best_tc2" << std::endl;
    
    std::ofstream detailed_csv("conv2d_profile_detailed.csv", std::ios::trunc);
    detailed_csv << "kernel,in_ch,out_ch,stride,pad,height,width,tc1,tc2,time_ms" << std::endl;
    
    std::cout << "Profiling Conv2D layers with thread tuning...\n" << std::endl;
    
    // All unique configurations from your file with actual dimensions
    profile_template<3, 1, 4, 1, 1>(480, 640, csv, detailed_csv);
    profile_template<3, 4, 8, 2, 1>(480, 640, csv, detailed_csv);
    profile_template<3, 8, 8, 1, 1>(240, 320, csv, detailed_csv);
    profile_template<3, 8, 24, 2, 1>(240, 320, csv, detailed_csv);
    profile_template<1, 1, 24, 1, 0>(120, 160, csv, detailed_csv);
    profile_template<3, 24, 24, 1, 1>(120, 160, csv, detailed_csv);
    profile_template<3, 24, 64, 2, 1>(120, 160, csv, detailed_csv);
    profile_template<3, 64, 64, 1, 1>(60, 80, csv, detailed_csv);
    profile_template<1, 64, 64, 1, 0>(60, 80, csv, detailed_csv);
    profile_template<3, 64, 64, 2, 1>(60, 80, csv, detailed_csv);
    profile_template<3, 64, 64, 1, 1>(30, 40, csv, detailed_csv);
    profile_template<3, 64, 128, 2, 1>(30, 40, csv, detailed_csv);
    profile_template<3, 128, 128, 1, 1>(15, 20, csv, detailed_csv);
    profile_template<1, 128, 64, 1, 0>(15, 20, csv, detailed_csv);
    profile_template<1, 64, 1, 1, 0>(60, 80, csv, detailed_csv);
    profile_template<1, 64, 65, 1, 0>(60, 80, csv, detailed_csv);
    
    csv.close();
    detailed_csv.close();
    std::cout << "\nResults saved to conv2d_profile.csv (summary) and conv2d_profile_detailed.csv (all configs)" << std::endl;
    
    return 0;
}