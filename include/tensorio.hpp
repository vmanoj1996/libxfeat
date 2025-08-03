#pragma once

#include <vector>
#include <string>
#include <hdf5.h>
#include <opencv2/opencv.hpp>

namespace tio
{
    inline void mkdir(const std::string& path) 
    {
        std::string command = "mkdir -p " + path;
        int result = system(command.c_str());

        if (result != 0) {
            throw std::runtime_error("Failed to create directory '" + path + "'. System command '" + command + "' returned " + std::to_string(result));
        }
    }

    inline void save_hdf5(const std::vector<float>& data, const std::vector<int>& shape, const std::string& filename, const std::string& dataset_name) 
    {
        std::vector<hsize_t> dims(shape.begin(), shape.end());
        
        hid_t file_id = H5Fcreate(filename.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
        hid_t space_id = H5Screate_simple(shape.size(), dims.data(), NULL);
        hid_t dataset_id = H5Dcreate2(file_id, dataset_name.c_str(), H5T_NATIVE_FLOAT, space_id,
                                    H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        
        H5Dwrite(dataset_id, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, data.data());
        H5Dflush(dataset_id);

        H5Fflush(file_id, H5F_SCOPE_GLOBAL); 

        
        H5Dclose(dataset_id);
        H5Sclose(space_id);
        H5Fclose(file_id);
    }

//     inline void save_hdf5(const std::vector<float>& data, const std::vector<int>& shape, const std::string& filename, const std::string& dataset_name) 
//     {
//     std::vector<hsize_t> dims(shape.begin(), shape.end());
    
//     hid_t file_id = H5Fcreate(filename.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
    
//     // Create property list for row-major (C-style) storage
//     hid_t dcpl = H5Pcreate(H5P_DATASET_CREATE);
//     std::vector<hsize_t> chunk_dims = dims;
//     H5Pset_chunk(dcpl, shape.size(), chunk_dims.data());
    
//     hid_t space_id = H5Screate_simple(shape.size(), dims.data(), NULL);
    
//     hid_t dataset_id = H5Dcreate2(file_id, dataset_name.c_str(), H5T_NATIVE_FLOAT, space_id,
//                                   H5P_DEFAULT, dcpl, H5P_DEFAULT);
    
//     H5Dwrite(dataset_id, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, data.data());
    
//     // Add row-major metadata attribute
//     hid_t attr_space = H5Screate(H5S_SCALAR);
//     hid_t attr_type = H5Tcopy(H5T_C_S1);
//     H5Tset_size(attr_type, 9);
//     hid_t attr = H5Acreate2(dataset_id, "memory_order", attr_type, attr_space, H5P_DEFAULT, H5P_DEFAULT);
//     H5Awrite(attr, attr_type, "row_major");
    
//     H5Aclose(attr);
//     H5Tclose(attr_type);
//     H5Sclose(attr_space);
//     H5Pclose(dcpl);
//     H5Dclose(dataset_id);
//     H5Sclose(space_id);
//     H5Fclose(file_id);
// }

    inline void save_yaml(const std::vector<float>& data, const std::vector<int>& shape,
                        const std::string& filename, const std::string& var_name) {
        cv::FileStorage fs(filename, cv::FileStorage::WRITE);
        
        if (shape.size() == 2) {
            cv::Mat mat(shape[0], shape[1], CV_32F, (void*)data.data());
            fs << var_name << mat;
        } else if (shape.size() == 3) {
            int sizes[] = {shape[0], shape[1], shape[2]};
            cv::Mat mat(3, sizes, CV_32F, (void*)data.data());
            fs << var_name << mat;
        }
        
        fs.release();
    }

}