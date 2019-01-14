//
// Created by wei on 1/14/19.
//

#include "MatrixCuda.h"
#include <Cuda/Common/UtilsCuda.h>

namespace open3d {

namespace cuda {

template<typename T>
MatrixCuda<T>::MatrixCuda() {
    max_rows_ = max_cols_ = pitch_ = -1;
}

template<typename T>
MatrixCuda<T>::MatrixCuda(int max_rows, int max_cols) {
    Create(max_rows, max_cols);
}

template<typename T>
MatrixCuda<T>::MatrixCuda(const MatrixCuda<T> &other) {
    server_ = other.server();
    max_rows_ = other.max_rows_;
    max_cols_ = other.max_cols_;
    pitch_ = other.pitch_;
}

template<typename T>
MatrixCuda<T> &MatrixCuda<T>::operator=(const MatrixCuda<T> &other) {
    if (this != &other) {
        Release();
        server_ = other.server();
        max_rows_ = other.max_rows_;
        max_cols_ = other.max_cols_;
        pitch_ = other.pitch_;
    }
    return *this;
}

template<typename T>
MatrixCuda<T>::~MatrixCuda() {
    Release();
}

template<typename T>
void MatrixCuda<T>::Create(int max_rows, int max_cols) {
    assert(max_rows > 0 && max_cols > 0);
    if (server_ != nullptr) {
        PrintError("[MatrixCuda]: Already created, abort!\n");
        return;
    }

    server_ = std::make_shared<MatrixCudaDevice<T>>();
    max_rows_ = max_rows;
    max_cols_ = max_cols;

    size_t
    pitch_size_t;
    CheckCuda(cudaMallocPitch(&server_->data_, &pitch_size_t,
                              sizeof(T) * max_cols_, max_rows_));
    pitch_ = (int) pitch_size_t;

    CheckCuda(cudaMalloc(&server_->iterator_rows_, sizeof(int)));
    CheckCuda(cudaMemset(server_->iterator_rows_, 0, sizeof(int)));

    UpdateServer();
}

template<typename T>
void MatrixCuda<T>::Release() {
    if (server_ != nullptr && server_.use_count() == 1) {
        CheckCuda(cudaFree(server_->data_));
        CheckCuda(cudaFree(server_->iterator_rows_));
    }

    server_ = nullptr;
    max_rows_ = max_cols_ = pitch_ = -1;
}

template<typename T>
void MatrixCuda<T>::UpdateServer() {
    assert(server_ != nullptr);
    server_->max_rows_ = max_rows_;
    server_->max_cols_ = max_cols_;
    server_->pitch_ = pitch_;
}

template<typename T>
void MatrixCuda<T>::Upload(
    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> &matrix) {
    assert(matrix.rows() == max_rows_ && matrix.cols() == max_cols_);

    CheckCuda(cudaMemcpy2D(server_->data_, pitch_,
                           matrix.data(), sizeof(T) * max_cols_,
                           sizeof(T) * max_cols_, max_rows_,
                           cudaMemcpyHostToDevice));
}

template<typename T>
Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
MatrixCuda<T>::Download() {
    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
        matrix(max_rows_, max_cols_);
    CheckCuda(cudaMemcpy2D(matrix.data(), sizeof(T) * max_cols_,
                           server_->data_, pitch_,
                           sizeof(T) * max_cols_, max_rows_,
                           cudaMemcpyDeviceToHost));
    return matrix;
}

template<typename T>
void MatrixCuda<T>::Fill(const T &val) {}

template<typename T>
void MatrixCuda<T>::Memset(int val) {
    CheckCuda(cudaMemset2D(server_->data_, pitch_, val, max_rows_, max_cols_));
}

template<typename T>
int MatrixCuda<T>::rows() const {
    int rows;
    CheckCuda(cudaMemcpy(&rows, server_->iterator_rows_, sizeof(int),
        cudaMemcpyDeviceToHost));
    return rows;
}

template<typename T>
void MatrixCuda<T>::set_iterator_rows(int row_position) {
    CheckCuda(cudaMemcpy(server_->iterator_rows_, &row_position, sizeof(int),
        cudaMemcpyHostToDevice));
}
}
}