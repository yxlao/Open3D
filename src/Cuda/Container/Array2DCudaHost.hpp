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
    device_ = other.device_;
    max_rows_ = other.max_rows_;
    max_cols_ = other.max_cols_;
    pitch_ = other.pitch_;
}

template<typename T>
Array2DCuda<T> &Array2DCuda<T>::operator=(const Array2DCuda<T> &other) {
    if (this != &other) {
        Release();
        device_ = other.device_;
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
    if (device_ != nullptr) {
        PrintError("[Array2DCuda]: Already created, abort!\n");
        return;
    }

    device_ = std::make_shared<Array2DCudaDevice<T>>();
    max_rows_ = max_rows;
    max_cols_ = max_cols;

    size_t pitch_size_t;
    CheckCuda(cudaMallocPitch(&device_->data_, &pitch_size_t,
                              sizeof(T) * max_cols_, max_rows_));
    pitch_ = (int) pitch_size_t;

    CheckCuda(cudaMalloc(&device_->iterator_rows_, sizeof(int)));
    CheckCuda(cudaMemset(device_->iterator_rows_, 0, sizeof(int)));

    UpdateDevice();
}

template<typename T>
void Array2DCuda<T>::Release() {
    if (device_ != nullptr && device_.use_count() == 1) {
        CheckCuda(cudaFree(device_->data_));
        CheckCuda(cudaFree(device_->iterator_rows_));
    }

    device_ = nullptr;
    max_rows_ = max_cols_ = pitch_ = -1;
}

template<typename T>
void Array2DCuda<T>::UpdateDevice() {
    assert(device_ != nullptr);
    device_->max_rows_ = max_rows_;
    device_->max_cols_ = max_cols_;
    device_->pitch_ = pitch_;
}

template<typename T>
void Array2DCuda<T>::CopyTo(Array2DCuda<T> &other) {
    assert(device_ != nullptr);

    if (this == &other) return;

    if (other.device_ == nullptr) {
        other.Create(max_rows_, max_cols_);
    } else if (other.max_rows_ < max_rows_ || other.max_cols_ < max_cols_) {
        PrintError("[Array2DCuda]: Dimension mismatch: (%d %d) vs (%d %d)\n",
                   other.max_rows_, other.max_cols_, max_rows_, max_cols_);
        return;
    }

    CheckCuda(cudaMemcpy2D(other.device_->data_, other.pitch_,
                           device_->data_, pitch_,
                           sizeof(T) * max_cols_, max_rows_,
                           cudaMemcpyDeviceToDevice));
}

template<typename T>
void Array2DCuda<T>::Upload(Eigen::Matrix<T, -1, -1, Eigen::RowMajor> &matrix) {
    if (device_ != nullptr) {
        assert(matrix.rows() == max_rows_ && matrix.cols() == max_cols_);
    } else {
        Create(matrix.rows(), matrix.cols());
    }

    CheckCuda(cudaMemcpy2D(device_->data_, pitch_,
                           matrix.data(), sizeof(T) * max_cols_,
                           sizeof(T) * max_cols_, max_rows_,
                           cudaMemcpyHostToDevice));
}

template<typename T>
Eigen::Matrix<T, -1, -1, Eigen::RowMajor> Array2DCuda<T>::Download() {
    Eigen::Matrix<T, -1, -1, Eigen::RowMajor> matrix(max_rows_, max_cols_);
    CheckCuda(cudaMemcpy2D(matrix.data(), sizeof(T) * max_cols_,
                           device_->data_, pitch_,
                           sizeof(T) * max_cols_, max_rows_,
                           cudaMemcpyDeviceToHost));
    return matrix;
}

template<typename T>
void Array2DCuda<T>::Fill(const T &val) {}

template<typename T>
void Array2DCuda<T>::Memset(int val) {
    CheckCuda(cudaMemset2D(device_->data_, pitch_, val,
        sizeof(T) * max_cols_, max_rows_));
}

template<typename T>
int Array2DCuda<T>::rows() const {
    int rows;
    CheckCuda(cudaMemcpy(&rows, device_->iterator_rows_, sizeof(int),
        cudaMemcpyDeviceToHost));
    return rows;
}

template<typename T>
void Array2DCuda<T>::set_iterator_rows(int row_position) {
    CheckCuda(cudaMemcpy(device_->iterator_rows_, &row_position, sizeof(int),
        cudaMemcpyHostToDevice));
}
}
}