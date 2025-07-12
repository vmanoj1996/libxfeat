#pragma once

#include <matx.h>

#include <cnpy.h>
#include <unordered_map>
#include <string>
#include <vector>

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
    std::unordered_map<std::string, cnpy::NpyArray> params_;
    
public:
    // Returns 0 on success, non-zero on error
    int loadParams(const std::string& filepath) {
        try {
            cnpy::npz_t npz_data = cnpy::npz_load(filepath);
            
            for (const auto& [name, array] : npz_data) {
                params_[name] = array;
            }
            
            return 0; // Success
        } catch (const std::exception& e) {
            return 1; // Error
        }
    }
    
    // Get parameter by name
    template<typename T>
    const T* getParam(const std::string& name, std::vector<size_t>& shape) const {
        auto it = params_.find(name);
        if (it == params_.end()) return nullptr;
        
        shape = it->second.shape;
        return it->second.data<T>();
    }
    
    // Get parameter shape
    std::vector<size_t> getShape(const std::string& name) const {
        auto it = params_.find(name);
        return (it != params_.end()) ? it->second.shape : std::vector<size_t>{};
    }
    
    // Check if parameter exists
    bool hasParam(const std::string& name) const {
        return params_.find(name) != params_.end();
    }
    
    size_t numParams() const { return params_.size(); }
};