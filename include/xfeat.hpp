#pragma once

#include <hdf5.h>
#include <unordered_map>
#include <string>
#include <vector>
#include <iostream>
#include <iomanip>
#include "matx.h"

#define FLOAT float

class XFeat
{

private:
    matx::cudaExecutor executor_;

    void loadParam();

public:
    XFeat(matx::cudaExecutor executor) : executor_(executor) {}
    XFeat(cudaStream_t stream = 0) : executor_(stream) {}

public:
    matx::tensor_t<FLOAT, 4> keypointHead();
    // descriptorHead();

};

class XFeatParams {
private:
    struct ParamData {
        std::vector<float> data;
        std::vector<size_t> shape;
    };
    
    std::unordered_map<std::string, ParamData> params_;
    std::unordered_map<std::string, ParamData> buffers_;
    
    // Helper to read dataset from HDF5
    bool readDataset(hid_t group_id, const std::string& name, ParamData& param) {
        hid_t dataset_id = H5Dopen2(group_id, name.c_str(), H5P_DEFAULT);
        if (dataset_id < 0) return false;
        
        hid_t space_id = H5Dget_space(dataset_id);
        int rank = H5Sget_simple_extent_ndims(space_id);
        
        std::vector<hsize_t> dims(rank);
        H5Sget_simple_extent_dims(space_id, dims.data(), nullptr);
        
        // Convert to size_t and calculate total size
        param.shape.resize(rank);
        size_t total_size = 1;
        for (int i = 0; i < rank; ++i) {
            param.shape[i] = static_cast<size_t>(dims[i]);
            total_size *= param.shape[i];
        }
        
        // Read data
        param.data.resize(total_size);
        herr_t status = H5Dread(dataset_id, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, 
                               H5P_DEFAULT, param.data.data());
        
        H5Sclose(space_id);
        H5Dclose(dataset_id);
        
        return status >= 0;
    }
    
    // Helper to print shape
    std::string shapeToString(const std::vector<size_t>& shape) const {
        if (shape.empty()) return "[]";
        
        std::string result = "[";
        for (size_t i = 0; i < shape.size(); ++i) {
            if (i > 0) result += ", ";
            result += std::to_string(shape[i]);
        }
        result += "]";
        return result;
    }
    
    // Helper function to get raw data and shape for a parameter
    std::pair<float*, std::vector<size_t>> getParamInfo(const std::string& name) {
        auto it = params_.find(name);
        if (it != params_.end()) {
            return {it->second.data.data(), it->second.shape};
        }
        
        auto buf_it = buffers_.find(name);
        if (buf_it != buffers_.end()) {
            return {buf_it->second.data.data(), buf_it->second.shape};
        }
        
        return {nullptr, {}};
    }
    
public:
    // Returns 0 on success, non-zero on error
    int loadParams(const std::string& filepath) {
        try {
            hid_t file_id = H5Fopen(filepath.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
            if (file_id < 0) return 1;
            
            // Read parameters group
            hid_t param_group = H5Gopen2(file_id, "parameters", H5P_DEFAULT);
            if (param_group >= 0) {
                hsize_t num_params;
                H5Gget_num_objs(param_group, &num_params);
                
                for (hsize_t i = 0; i < num_params; ++i) {
                    char name[256];
                    H5Gget_objname_by_idx(param_group, i, name, sizeof(name));
                    
                    ParamData param;
                    if (readDataset(param_group, name, param)) {
                        params_[std::string(name)] = std::move(param);
                    }
                }
                H5Gclose(param_group);
            }
            
            // Read buffers group
            hid_t buffer_group = H5Gopen2(file_id, "buffers", H5P_DEFAULT);
            if (buffer_group >= 0) {
                hsize_t num_buffers;
                H5Gget_num_objs(buffer_group, &num_buffers);
                
                for (hsize_t i = 0; i < num_buffers; ++i) {
                    char name[256];
                    H5Gget_objname_by_idx(buffer_group, i, name, sizeof(name));
                    
                    ParamData buffer;
                    if (readDataset(buffer_group, name, buffer)) {
                        buffers_[std::string(name)] = std::move(buffer);
                    }
                }
                H5Gclose(buffer_group);
            }
            
            H5Fclose(file_id);
            return 0; // Success
            
        } catch (const std::exception& e) {
            return 1; // Error
        }
    }
    
    // Helper functions to construct tensors of specific ranks

    matx::tensor_t<float, 0> makeParam0D(const std::string& name) 
    {
        auto [data, shape] = getParamInfo(name);
        if (data && shape.size() == 0) {
            return matx::make_tensor<float>(data, {}, false);
        }
        // Return empty 0D tensor if not found or wrong rank
        return matx::make_tensor<float>(nullptr, {}, false);
    }
    matx::tensor_t<float, 1> makeParam1D(const std::string& name) 
    {
        auto [data, shape] = getParamInfo(name);
        if (data && shape.size() == 1) {
            cuda::std::array<matx::index_t, 1> matx_shape{static_cast<matx::index_t>(shape[0])};
            return matx::make_tensor<float>(data, matx_shape, false);
        }
        // Return empty tensor if not found or wrong rank
        cuda::std::array<matx::index_t, 1> empty_shape{0};
        return matx::make_tensor<float>(nullptr, empty_shape, false);
    }

    matx::tensor_t<float, 2> makeParam2D(const std::string& name) 
    {
        auto [data, shape] = getParamInfo(name);
        if (data && shape.size() == 2) {
            cuda::std::array<matx::index_t, 2> matx_shape{static_cast<matx::index_t>(shape[0]), 
                                                          static_cast<matx::index_t>(shape[1])};
            return matx::make_tensor<float>(data, matx_shape, false);
        }
        // Return empty tensor if not found or wrong rank
        cuda::std::array<matx::index_t, 2> empty_shape{0, 0};
        return matx::make_tensor<float>(nullptr, empty_shape, false);
    }

    matx::tensor_t<float, 3> makeParam3D(const std::string& name) 
    {
        auto [data, shape] = getParamInfo(name);
        if (data && shape.size() == 3) {
            cuda::std::array<matx::index_t, 3> matx_shape{static_cast<matx::index_t>(shape[0]), 
                                                          static_cast<matx::index_t>(shape[1]),
                                                          static_cast<matx::index_t>(shape[2])};
            return matx::make_tensor<float>(data, matx_shape, false);
        }
        // Return empty tensor if not found or wrong rank
        cuda::std::array<matx::index_t, 3> empty_shape{0, 0, 0};
        return matx::make_tensor<float>(nullptr, empty_shape, false);
    }

    matx::tensor_t<float, 4> makeParam4D(const std::string& name) 
    {
        auto [data, shape] = getParamInfo(name);
        if (data && shape.size() == 4) {
            cuda::std::array<matx::index_t, 4> matx_shape{static_cast<matx::index_t>(shape[0]), 
                                                          static_cast<matx::index_t>(shape[1]),
                                                          static_cast<matx::index_t>(shape[2]),
                                                          static_cast<matx::index_t>(shape[3])};
            return matx::make_tensor<float>(data, matx_shape, false);
        }
        // Return empty tensor if not found or wrong rank
        cuda::std::array<matx::index_t, 4> empty_shape{0, 0, 0, 0};
        return matx::make_tensor<float>(nullptr, empty_shape, false);
    }
    
    // Get parameter shape
    std::vector<size_t> getShape(const std::string& name) const 
    {
        auto it = params_.find(name);
        if (it != params_.end()) return it->second.shape;
        
        auto buf_it = buffers_.find(name);
        if (buf_it != buffers_.end()) return buf_it->second.shape;
        
        return std::vector<size_t>{};
    }
    
    // Print all parameters and their shapes
    void printParams() const 
    {
        std::cout << "\n=== PARAMETERS ===\n";
        for (const auto& [name, param] : params_) {
            std::cout << std::setw(50) << std::left << name 
                      << " : " << shapeToString(param.shape) << "\n";
        }
        
        std::cout << "\n=== BUFFERS ===\n";
        for (const auto& [name, buffer] : buffers_) {
            std::cout << std::setw(50) << std::left << name 
                      << " : " << shapeToString(buffer.shape) << "\n";
        }
        
        std::cout << "\nTotal: " << params_.size() << " parameters, " 
                  << buffers_.size() << " buffers\n";
    }
    
    // Check if parameter exists
    bool hasParam(const std::string& name) const 
    {
        return params_.find(name) != params_.end() || 
               buffers_.find(name) != buffers_.end();
    }
    
    size_t numParams() const { return params_.size() + buffers_.size(); }
};