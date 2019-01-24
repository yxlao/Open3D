//
// Created by wei on 1/14/19.
//

#include "Array2DCuda.h"
#include <Cuda/Common/UtilsCuda.h>

namespace open3d {

namespace cuda {

template<typename T>
Array2DCuda<T>::Array2DCuda() {
    max_rows_ = max_cols_ = pitch_ = -1;
}

template<typename T>
Array2DCuda<T>::Array2DCuda(int max_rows, int max_cols) {
    Create(max_rows, max_cols);
}

template<typename T>
Array2DCuda<T>::Array2DCuda(const Array2DCuda<T> &other) {
    server_ = other.server();
    max_rows_ = other.max_rows_;
    max_cols_ = other.max_cols_;
    pitch_ = other.pitch_;
}

template<typename T>
Array2DCuda<T> &Array2DCuda<T>::operator=(const Array2DCuda<T> &other) {
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
Array2DCuda<T>::~Array2DCuda() {
    Release();
}

template<typename T>
void Array2DCuda<T>::Create(int max_rows, int max_cols) {
    assert(max_rows > 0 && max_cols > 0);
    if (server_ != nullptr) {
        PrintError("[Array2DCuda]: Already created, abort!\n");
        return;
    }

    server_ = std::make_shared<Array2DCudaDevice<T>>();
    max_rows_ = max_rows;
    max_cols_ = max_cols;

    size_t pitch_size_t;
    CheckCuda(cudaMallocPitch(&server_->data_, &pitch_size_t,
                              sizeof(T) * max_cols_, max_rows_));
    pitch_ = (int) pitch_size_t;

    CheckCuda(cudaMalloc(&server_->iterator_rows_, sizeof(int)));
    CheckCuda(cudaMemset(server_->iterator_rows_, 0, sizeof(int)));

    UpdateServer();
}

template<typename T>
void Array2DCuda<T>::Release() {
    if (server_ != nullptr && server_.use_count() == 1) {
        CheckCuda(cudaFree(server_->data_));
        CheckCuda(cudaFree(server_->iterator_rows_));
    }

    server_ = nullptr;
    max_rows_ = max_cols_ = pitch_ = -1;
}

template<typename T>
void Array2DCuda<T>::UpdateServer() {
    assert(server_ != nullptr);
    server_->max_rows_ = max_rows_;
    server_->max_cols_ = max_cols_;
    server_->pitch_ = pitch_;
}

template<typename T>
void Array2DCuda<T>::Upload(Eigen::Matrix<T, -1, -1, Eigen::RowMajor> &matrix) {
    if (server_ != nullptr) {
        assert(matrix.rows() == max_rows_ && matrix.cols() == max_cols_);
    } else {
        Create(matrix.rows(), matrix.cols());
    }

    CheckCuda(cudaMemcpy2D(server_->data_, pitch_,
                           matrix.data(), sizeof(T) * max_cols_,
                           sizeof(T) * max_cols_, max_rows_,
                           cudaMemcpyHostToDevice));
}

template<typename T>
Eigen::Matrix<T, -1, -1, Eigen::RowMajor> Array2DCuda<T>::Download() {
    Eigen::Matrix<T, -1, -1, Eigen::RowMajor> matrix(max_rows_, max_cols_);
    CheckCuda(cudaMemcpy2D(matrix.data(), sizeof(T) * max_cols_,
                           server_->data_, pitch_,
                           sizeof(T) * max_cols_, max_rows_,
                           cudaMemcpyDeviceToHost));
    return matrix;
}

template<typename T>
void Array2DCuda<T>::Fill(const T &val) {}

template<typename T>
void Array2DCuda<T>::Memset(int val) {
    CheckCuda(cudaMemset2D(server_->data_, pitch_, val,
        sizeof(T) * max_cols_, max_rows_));
}

template<typename T>
int Array2DCuda<T>::rows() const {
    int rows;
    CheckCuda(cudaMemcpy(&rows, server_->iterator_rows_, sizeof(int),
        cudaMemcpyDeviceToHost));
    return rows;
}

template<typename T>
void Array2DCuda<T>::set_iterator_rows(int row_position) {
    CheckCuda(cudaMemcpy(server_->iterator_rows_, &row_position, sizeof(int),
        cudaMemcpyHostToDevice));
}
}
}