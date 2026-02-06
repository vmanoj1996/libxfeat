# libxfeat - XFeat Feature Extractor written in native CUDA and C++


## Build instructions
```
mkdir build 
cd build

cmake .. && make -j && ctest --verbose

# some tests require a python environment containing torch or numpy etc. Edit the below lines in CMakeLists.txt
# source ~/miniconda3/etc/profile.d/conda.sh && conda activate cusfm


# For debug builds, use cmake -DCMAKE_BUILD_TYPE=Debug ..



```

### Comparison against Pytorch implementation

```
see perf_xfeat.py
--- Performance Summary (original implementation on 4070) ---
10: Total time for 1000 runs: 1000.760 ms
10: Average latency:        1.001 ms
10: Average throughput (FPS): 999.241
10: Median latency:         0.985 ms
10: Minimum latency:        0.964 ms
10: Maximum latency:        2.113 ms
---------------------------

--- Performance Summary (This implementation with TF32 multiplications enabled (tensor cores) on 4070) ---
cmake -D USE_TF32=ON ..
8: Total time for 1000 runs: 670.281 ms
8: Average latency:        0.670 ms
8: Average throughput (FPS): 1491.912
8: Median latency:         0.668 ms
8: Minimum latency:        0.645 ms
8: Maximum latency:        1.564 ms
8: Mean,var: 0.670 ± 0.040 ms

---------------------------

--- Performance Summary (This implementation with full FP32 on 4070) ---
cmake -D USE_TF32=OFF ..
8: Total time for 1000 runs: 757.521 ms
8: Average latency:        0.758 ms
8: Average throughput (FPS): 1320.095
8: Median latency:         0.754 ms
8: Minimum latency:        0.727 ms
8: Maximum latency:        1.623 ms
8: Mean,var: 0.758 ± 0.045 ms
---------------------------
```

## Sample Heatmap output
<img width="240" height="135" alt="TajMahal" src="https://github.com/user-attachments/assets/cab1eefc-83d2-42f5-a9a2-ccc6e1deb228" />
<img width="240" height="135" alt="heatmap_visualization" src="https://github.com/user-attachments/assets/b86fa963-2862-4543-8155-ffd9d4a2998f" />

## Citation
If you find this code useful for your research, please cite the original paper along with this repo:

```
@software{Velmurugan_libxfeat_2025,
  author = {Velmurugan, Manoj},
  title = {{libxfeat: A C++/CUDA Implementation of XFeat}},
  url = {https://github.com/vmanoj1996/libxfeat},
  year = {2025}
}
```

```
@INPROCEEDINGS{potje2024cvpr,
  author={Potje, Guilherme and Cadar, Felipe and Araujo, André and Martins, Renato and Nascimento, Erickson R.},
  booktitle={2024 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)}, 
  title={XFeat: Accelerated Features for Lightweight Image Matching}, 
  year={2024},
  pages={2682-2691},
  keywords={Visualization;Accuracy;Image matching;Pose estimation;Feature extraction;Hardware;Real-time systems;Image matching;Local features;Lightweight;Fast},
  doi={10.1109/CVPR52733.2024.00259}}
```
https://github.com/verlab/accelerated_features

