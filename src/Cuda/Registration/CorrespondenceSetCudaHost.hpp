//
// Created by wei on 1/14/19.
//

#include "CorrespondenceSetCuda.h"
#include <Core/Core.h>

namespace open3d {
namespace cuda {

CorrespondenceSetCuda::CorrespondenceSetCuda(
    const CorrespondenceSetCuda &other) {
    server_ = other.server_;

    matrix_ = other.matrix_;
    indices_ = other.indices_;
    nn_count_ = other.nn_count_;
}

CorrespondenceSetCuda& CorrespondenceSetCuda::operator=(
    const CorrespondenceSetCuda &other) {
    if (this != &other) {
        server_ = other.server_;

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
    if (server_ != nullptr) {
        PrintError("[CorrespondenceSetCuda] Already created, abort.\n");
        return;
    }

    server_ = std::make_shared<CorrespondenceSetCudaDevice>();
    matrix_.Create(max_rows, max_cols);
    indices_.Create(max_cols);
    nn_count_.Create(max_cols);
}

void CorrespondenceSetCuda::Release() {
    if (server_ != nullptr && server_.use_count() == 1) {
        matrix_.Release();
        indices_.Release();
        nn_count_.Release();
    }

    server_ = nullptr;
}

void CorrespondenceSetCuda::SetCorrespondenceMatrix(
    Eigen::MatrixXi &corres_matrix) {

    Eigen::Matrix<int, -1, -1, Eigen::RowMajor>
        corres_matrix_rowmajor = corres_matrix;

    if (server_ == nullptr) {
        Create(corres_matrix.rows(), corres_matrix.cols());
    } else {
        assert(corres_matrix.rows() == matrix_.max_rows_
                   && corres_matrix.cols() == matrix_.max_cols_);
    }

    matrix_.Upload(corres_matrix_rowmajor);
    indices_.Resize(matrix_.max_cols_);
    nn_count_.Resize(matrix_.max_cols_);
    UpdateServer();
}

void CorrespondenceSetCuda::Compress() {
    assert(matrix_.max_cols_ > 0);

    CorrespondenceSetCudaKernelCaller::
    CompressCorrespondenceKernelCaller(*this);
}


void CorrespondenceSetCuda::UpdateServer() {
    assert(server_ != nullptr);

    server_->matrix_ = *matrix_.server();
    server_->indices_ = *indices_.server();
    server_->nn_count_ = *nn_count_.server();
}

}
}