//
// Created by wei on 1/14/19.
//

#pragma once

#include <Cuda/Container/Array2DCuda.h>
#include <Cuda/Container/ArrayCuda.h>

namespace open3d {
namespace cuda {

class CorrespondenceSetCudaDevice {
public:
    Array2DCudaDevice<int> matrix_;
    ArrayCudaDevice<int> indices_;
};

class CorrespondenceSetCuda {
private:
    std::shared_ptr<CorrespondenceSetCudaDevice> server_ = nullptr;

public:
    CorrespondenceSetCuda() {};
    CorrespondenceSetCuda(const CorrespondenceSetCuda &other);
    CorrespondenceSetCuda& operator=(const CorrespondenceSetCuda &other);
    ~CorrespondenceSetCuda();

    void Create();
    void Release();

    Array2DCuda<int> matrix_;

    /* Row indices of correpsondences (no need to be ordered) */
    ArrayCuda<int> indices_;

    void SetCorrespondenceMatrix(
        Eigen::Matrix<int, -1, -1, Eigen::RowMajor> &corres_matrix);

    void Compress();

    void UpdateServer();

public:
    std::shared_ptr<CorrespondenceSetCudaDevice> server() {
        return server_;
    }
    const std::shared_ptr<CorrespondenceSetCudaDevice> server() const {
        return server_;
    }
};

class CorrespondenceSetCudaKernelCaller {
public:
    static void CompressCorrespondenceKernelCaller(
        CorrespondenceSetCuda& corres);
};

__GLOBAL__
void CompressCorrespondencesKernel(CorrespondenceSetCudaDevice corres);
}
}


