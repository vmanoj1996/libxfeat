# libxfeat - XFeat Feature Extractor written in native CUDA and C++

```
mkdir build 
cd build

cmake .. && make -j && ctest --verbose

# some tests require a python environment containing torch or numpy etc. Edit the below lines in CMakeLists.txt
# source ~/miniconda3/etc/profile.d/conda.sh && conda activate cusfm

```

## Sample output
<img width="240" height="135" alt="TajMahal" src="https://github.com/user-attachments/assets/cab1eefc-83d2-42f5-a9a2-ccc6e1deb228" />
<img width="240" height="135" alt="heatmap_visualization" src="https://github.com/user-attachments/assets/b86fa963-2862-4543-8155-ffd9d4a2998f" />


## Original Reference

https://github.com/verlab/accelerated_features
