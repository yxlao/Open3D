//
// Created by wei on 1/14/19.
//

#pragma once

#include <Cuda/Container/MatrixCuda.h>
#include <Cuda/Container/ArrayCuda.h>

namespace open3d {
namespace cuda {

class CorrespondenceSetCuda {
public:
    MatrixCuda<int> corres_matrix_;

    /* Row indices of correpsondences (no need to be ordered) */
    ArrayCuda<int> corres_indices_;

public:
    void SetCorrespondenceMatrix(
        Eigen::Matrix<int, -1, -1, Eigen::RowMajor> &corres_matrix) {
        corres_matrix_.Upload(corres_matrix);

        corres_indices_.Resize(corres_matrix_.max_rows_);
    }
    void Compress();

};

class CorrespondenceSetCudaKernelCaller {
public:
    static void CompressCorrespondenceKernelCaller(
        MatrixCuda<int> &corres_matrix,
        ArrayCuda<int> &corres_indices);
};

__GLOBAL__
void CompressCorrespondencesKernel(
    MatrixCudaDevice<int> corres_matrix,
    ArrayCuda<int> corres_indices);
}
}


