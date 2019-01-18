//
// Created by wei on 1/14/19.
//

#include "CorrespondenceSetCuda.h"
#include <Core/Core.h>

namespace open3d {
namespace cuda {

CorrespondenceSetCuda::CorrespondenceSetCuda(
    const CorrespondenceSetCuda &other) {
    server_ = other.server();

    matrix_ = other.matrix_;
    indices_ = other.indices_;
}

CorrespondenceSetCuda& CorrespondenceSetCuda::operator=(
    const CorrespondenceSetCuda &other) {
    if (this != &other) {
        server_ = other.server();

        matrix_ = other.matrix_;
        indices_ = other.indices_;
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
    indices_.Create(max_rows);
}

void CorrespondenceSetCuda::Release() {
    if (server_ != nullptr && server_.use_count() == 1) {
        matrix_.Release();
        indices_.Release();
    }

    server_ = nullptr;
}

void CorrespondenceSetCuda::SetCorrespondenceMatrix(
    Eigen::Matrix<int, -1, -1, Eigen::RowMajor> &corres_matrix) {
    if (server_ == nullptr) {
        Create(corres_matrix.rows(), corres_matrix.cols());
    }

    matrix_.Upload(corres_matrix);
    indices_.Resize(matrix_.max_rows_);
    UpdateServer();
}

void CorrespondenceSetCuda::Compress() {
    assert(matrix_.max_rows_ > 0);

    CorrespondenceSetCudaKernelCaller::
    CompressCorrespondenceKernelCaller(*this);
}


void CorrespondenceSetCuda::UpdateServer() {
    assert(server_ != nullptr);

    server_->matrix_ = *matrix_.server();
    server_->indices_ = *indices_.server();
}

}
}