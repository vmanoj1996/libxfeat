# libxfeat
XFeat Feature Extractor ported to native CUDA and C++

```
mkdir build 
cd build

cmake .. && make -j && ctest --verbose

# some tests require python to run checkout these kinds of lines in cmakelists
# source ~/miniconda3/etc/profile.d/conda.sh && conda activate cusfm

```
