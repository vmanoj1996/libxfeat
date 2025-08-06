# libxfeat - XFeat Feature Extractor written in native CUDA and C++

```
mkdir build 
cd build

cmake .. && make -j && ctest --verbose

# some tests require a python environment containing torch or numpy etc. Edit the below lines in CMakeLists.txt
# source ~/miniconda3/etc/profile.d/conda.sh && conda activate cusfm


# For debug builds, use cmake -DCMAKE_BUILD_TYPE=Debug ..


```

## Sample output
<img width="240" height="135" alt="TajMahal" src="https://github.com/user-attachments/assets/cab1eefc-83d2-42f5-a9a2-ccc6e1deb228" />
<img width="240" height="135" alt="heatmap_visualization" src="https://github.com/user-attachments/assets/b86fa963-2862-4543-8155-ffd9d4a2998f" />

## Citation
If you find this code useful for your research, please cite the original paper along with this repo:

https://github.com/verlab/accelerated_features
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


