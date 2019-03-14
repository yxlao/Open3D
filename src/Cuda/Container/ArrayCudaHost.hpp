//
// Created by wei on 11/9/18.
//

#pragma once

#include "ArrayCuda.h"
#include <src/Cuda/Common/UtilsCuda.h>
#include <src/Cuda/Common/LinearAlgebraCuda.h>

#include <Open3D/Utility/Console.h>
#include <cuda_runtime.h>

namespace open3d {

namespace cuda {
template<typename T>
ArrayCuda<T>::ArrayCuda() {
    max_capacity_ = -1;
}

template<typename T>
ArrayCuda<T>::ArrayCuda(int max_capacity) {
    Create(max_capacity);
}

template<typename T>
ArrayCuda<T>::ArrayCuda(const ArrayCuda<T> &other) {
    device_ = other.device_;
    max_capacity_ = other.max_capacity_;
}

template<typename T>
ArrayCuda<T> &ArrayCuda<T>::operator=(const ArrayCuda<T> &other) {
    if (this != &other) {
        Release();

        device_ = other.device_;
        max_capacity_ = other.max_capacity_;
    }
    return *this;
}

template<typename T>
ArrayCuda<T>::~ArrayCuda() {
    Release();
}

template<typename T>
void ArrayCuda<T>::Resize(int max_capacity) {
    if (device_ == nullptr) {
        Create(max_capacity);
    } else {
        if (max_capacity_ == max_capacity) return;

        assert(max_capacity_ < max_capacity);

        T *resized_data;
        CheckCuda(cudaMalloc(&(resized_data), sizeof(T) * max_capacity));

        int used_data_count = size();
        CheckCuda(cudaMemcpy(resized_data, device_->data_,
                             sizeof(T) * used_data_count,
                             cudaMemcpyDeviceToDevice));
        CheckCuda(cudaFree(device_->data_));
        device_->data_ = resized_data;

        max_capacity_ = max_capacity;
        device_->max_capacity_ = max_capacity;
    }
}

template<typename T>
void ArrayCuda<T>::Create(int max_capacity) {
    assert(max_capacity > 0);

    if (device_ != nullptr) {
        utility::PrintError("[ArrayCuda]: Already created, abort!\n");
        return;
    }

    device_ = std::make_shared<ArrayCudaDevice<T >>();
    max_capacity_ = max_capacity;
    device_->max_capacity_ = max_capacity;

    CheckCuda(cudaMalloc(&(device_->data_), sizeof(T) * max_capacity));
    CheckCuda(cudaMemset(device_->data_, 0, sizeof(T) * max_capacity));
    CheckCuda(cudaMalloc(&(device_->iterator_), sizeof(int)));
    CheckCuda(cudaMemset(device_->iterator_, 0, sizeof(int)));
}

template<typename T>
void ArrayCuda<T>::Release() {
    if (device_ != nullptr && device_.use_count() == 1) {
        CheckCuda(cudaFree(device_->data_));
        CheckCuda(cudaFree(device_->iterator_));
    }

    device_ = nullptr;
    max_capacity_ = -1;
}

template<typename T>
void ArrayCuda<T>::CopyFromDeviceArray(const T *array, int size) {
    if (device_ == nullptr) {
        Create(size);
    } else if (device_->max_capacity_ < size) {
        utility::PrintError("[ArrayCuda]: max capacity %d < %d, abort!\n",
                   max_capacity_, size);
        return;
    }
    CheckCuda(cudaMemcpy(device_->data_, array,
                         sizeof(T) * size,
                         cudaMemcpyDeviceToDevice));

}

template<typename T>
void ArrayCuda<T>::CopyTo(ArrayCuda<T> &other) const {
    assert(device_ != nullptr);

    if (this == &other) return;

    if (other.device_ == nullptr) {
        other.Create(max_capacity_);
    } else if (other.max_capacity_ < max_capacity_) {
        utility::PrintError("[ArrayCuda]: other.max_capacity %d < %d, abort!\n",
                   max_capacity_, other.max_capacity_);
        return;
    }

    CheckCuda(cudaMemcpy(other.device_->data(), device_->data_,
                         sizeof(T) * max_capacity_,
                         cudaMemcpyDeviceToDevice));
}

template<typename T>
void ArrayCuda<T>::Upload(std::vector<T> &data) {
    assert(device_ != nullptr);

    int size = data.size();
    assert(size <= max_capacity_);
    CheckCuda(cudaMemcpy(device_->data_, data.data(),
                         sizeof(T) * size,
                         cudaMemcpyHostToDevice));
    CheckCuda(cudaMemcpy(device_->iterator_, &size,
                         sizeof(int),
                         cudaMemcpyHostToDevice));
}

template<typename T>
void ArrayCuda<T>::Upload(const T *data, int size) {
    assert(device_ != nullptr);

    assert(size <= max_capacity_);
    CheckCuda(cudaMemcpy(device_->data_, data,
                         sizeof(T) * size,
                         cudaMemcpyHostToDevice));
    CheckCuda(cudaMemcpy(device_->iterator_, &size,
                         sizeof(int),
                         cudaMemcpyHostToDevice));
}

template<class T>
std::vector<T> ArrayCuda<T>::Download() {
    assert(device_ != nullptr);

    std::vector<T> ret;
    int iterator_count = size();
    ret.resize(iterator_count);

    if (iterator_count == 0) return ret;

    CheckCuda(cudaMemcpy(ret.data(), device_->data_,
                         sizeof(T) * iterator_count,
                         cudaMemcpyDeviceToHost));

    return ret;
}

template<typename T>
std::vector<T> ArrayCuda<T>::DownloadAll() {
    assert(device_ != nullptr);

    std::vector<T> ret;
    ret.resize(max_capacity_);

    CheckCuda(cudaMemcpy(ret.data(), device_->data_,
                         sizeof(T) * max_capacity_,
                         cudaMemcpyDeviceToHost));

    return ret; /* RVO will handle this (hopefully) */
}

template<typename T>
void ArrayCuda<T>::Fill(const T &val) {
    if (device_ == nullptr) return;
    ArrayCudaKernelCaller<T>::Fill(*this, val);
}

template<typename T>
void ArrayCuda<T>::Memset(int val) {
    assert(device_ != nullptr);

    CheckCuda(cudaMemset(device_->data_, val, sizeof(T) * max_capacity_));
}

template<class T>
void ArrayCuda<T>::Clear() {
    assert(device_ != nullptr);

    CheckCuda(cudaMemset(device_->iterator_, 0, sizeof(int)));
}

template<class T>
int ArrayCuda<T>::size() const {
    assert(device_ != nullptr);

    int ret;
    CheckCuda(cudaMemcpy(&ret, device_->iterator_,
                         sizeof(int),
                         cudaMemcpyDeviceToHost));
    return ret;
}

template<typename T>
void ArrayCuda<T>::set_iterator(int iterator_position) {
    assert(device_ != nullptr);

    assert(0 <= iterator_position && iterator_position <= max_capacity_);
    CheckCuda(cudaMemcpy(device_->iterator_, &iterator_position,
                         sizeof(int),
                         cudaMemcpyHostToDevice));
}
} // cuda
} // open3d