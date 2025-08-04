# libxfeat
XFeat Feature Extractor ported to native CUDA and C++

```
mkdir build 
cd build

cmake .. && make -j && ctest --verbose

# some tests require a python environment containing torch or numpy etc. Edit the below lines in CMakeLists.txt
# source ~/miniconda3/etc/profile.d/conda.sh && conda activate cusfm

```
