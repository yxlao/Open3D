//
// Created by wei on 1/14/19.
//

#include "CorrespondenceSetCuda.h"
#include <Core/Core.h>

namespace open3d {
namespace cuda {

CorrespondenceSetCuda::CorrespondenceSetCuda(
    const CorrespondenceSetCuda &other) {
    device_ = other.device_;

    matrix_ = other.matrix_;
    indices_ = other.indices_;
    nn_count_ = other.nn_count_;
}

CorrespondenceSetCuda& CorrespondenceSetCuda::operator=(
    const CorrespondenceSetCuda &other) {
    if (this != &other) {
        device_ = other.device_;

        matrix_ = other.matrix_;
        indices_ = other.indices_;
        nn_count_ = other.nn_count_;
    }

    return *this;
}

CorrespondenceSetCuda::~CorrespondenceSetCuda() {
    Release();
}

void CorrespondenceSetCuda::Create(int max_rows, int max_cols) {
    if (device_ != nullptr) {
        PrintError("[CorrespondenceSetCuda] Already created, abort.\n");
        return;
    }

    device_ = std::make_shared<CorrespondenceSetCudaDevice>();
    matrix_.Create(max_rows, max_cols);
    indices_.Create(max_cols);
    nn_count_.Create(max_cols);
}

void CorrespondenceSetCuda::Release() {
    if (device_ != nullptr && device_.use_count() == 1) {
        matrix_.Release();
        indices_.Release();
        nn_count_.Release();
    }

    device_ = nullptr;
}

void CorrespondenceSetCuda::SetCorrespondenceMatrix(
    Eigen::MatrixXi &corres_matrix) {

    Eigen::Matrix<int, -1, -1, Eigen::RowMajor>
        corres_matrix_rowmajor = corres_matrix;

    if (device_ == nullptr) {
        Create(corres_matrix.rows(), corres_matrix.cols());
    } else {
        assert(corres_matrix.rows() == matrix_.max_rows_
                   && corres_matrix.cols() == matrix_.max_cols_);
    }

    matrix_.Upload(corres_matrix_rowmajor);
    indices_.Resize(matrix_.max_cols_);
    nn_count_.Resize(matrix_.max_cols_);
    UpdateDevice();
}

void CorrespondenceSetCuda::SetCorrespondenceMatrix(
    Array2DCuda<int> &corres_matrix) {
    if (device_ == nullptr) {
        Create(corres_matrix.max_rows_, corres_matrix.max_cols_);
    } else {
        assert(corres_matrix.max_rows_ == matrix_.max_rows_);
        assert(corres_matrix.max_cols_ == matrix_.max_cols_);
    }

    corres_matrix.CopyTo(matrix_);
    indices_.Resize(matrix_.max_cols_);
    nn_count_.Resize(matrix_.max_cols_);
    UpdateDevice();
}

void CorrespondenceSetCuda::Compress() {
    assert(matrix_.max_cols_ > 0);

    CorrespondenceSetCudaKernelCaller::CompressCorrespondence(*this);
}


void CorrespondenceSetCuda::UpdateDevice() {
    assert(device_ != nullptr);

    device_->matrix_ = *matrix_.device_;
    device_->indices_ = *indices_.device_;
    device_->nn_count_ = *nn_count_.device_;
}

}
}