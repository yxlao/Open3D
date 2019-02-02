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
    ArrayCudaDevice<int> nn_count_;
};

class CorrespondenceSetCuda {
public:
    std::shared_ptr<CorrespondenceSetCudaDevice> device_ = nullptr;

public:
    CorrespondenceSetCuda() { device_ = nullptr; };
    CorrespondenceSetCuda(const CorrespondenceSetCuda &other);
    CorrespondenceSetCuda& operator=(const CorrespondenceSetCuda &other);
    ~CorrespondenceSetCuda();

    void Create(int max_rows, int max_cols);
    void Release();
    void UpdateDevice();

    /* Dimensions: correspondences x queries */
    Array2DCuda<int> matrix_;

    /* Row indices of correpsondences (no need to be ordered) */
    ArrayCuda<int> indices_;
    ArrayCuda<int> nn_count_;

    void SetCorrespondenceMatrix(Eigen::MatrixXi &corres_matrix);
    void SetCorrespondenceMatrix(Array2DCuda<int> &corres_matrix);
    void Compress();
};

class CorrespondenceSetCudaKernelCaller {
public:
    static void CompressCorrespondence(
        CorrespondenceSetCuda& corres);
};

__GLOBAL__
void CompressCorrespondencesKernel(CorrespondenceSetCudaDevice corres);
}
}


