#pragma once

#include <hdf5.h>
#include <unordered_map>
#include <string>
#include <vector>
#include <iostream>
#include <iomanip>
#include "primitives.hpp"


class XFeatParams {
private:
   struct ParamData {
       std::vector<FLOAT> data;
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
       
       // Read data as FLOAT
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
   
public:
    XFeatParams(const std::string& filepath)
    {
        if (loadParams(filepath) != 0)
        {
            throw std::runtime_error("Failed to load parameters from file: " + filepath);
        }
        else
        {   
            std::cout<<"Model Parameter Load Success\n";
        }
    }

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
   
   // Get parameter data as vector
   std::vector<FLOAT> getParam(const std::string& name) const {
       auto it = params_.find(name);
       if (it != params_.end()) {
           return it->second.data;
       }
       
       auto buf_it = buffers_.find(name);
       if (buf_it != buffers_.end()) {
           return buf_it->second.data;
       }
       
       return std::vector<FLOAT>{}; // Empty vector if not found
   }
   
   // Get parameter shape
   std::vector<size_t> getShape(const std::string& name) const {
       auto it = params_.find(name);
       if (it != params_.end()) return it->second.shape;
       
       auto buf_it = buffers_.find(name);
       if (buf_it != buffers_.end()) return buf_it->second.shape;
       
       return std::vector<size_t>{};
   }
   
   // Get both data and shape together
   std::pair<std::vector<FLOAT>, std::vector<size_t>> getParamWithShape(const std::string& name) const {
       auto it = params_.find(name);
       if (it != params_.end()) {
           return {it->second.data, it->second.shape};
       }
       
       auto buf_it = buffers_.find(name);
       if (buf_it != buffers_.end()) {
           return {buf_it->second.data, buf_it->second.shape};
       }
       
       return {std::vector<FLOAT>{}, std::vector<size_t>{}};
   }
   
   // Print all parameters and their shapes
   void printParams() const {
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
   bool hasParam(const std::string& name) const {
       return params_.find(name) != params_.end() || 
              buffers_.find(name) != buffers_.end();
   }
   
   size_t numParams() const { return params_.size() + buffers_.size(); }
};