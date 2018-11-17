//
// Created by wei on 11/9/18.
//

#pragma once

#include "ArrayCuda.h"
#include <Cuda/Common/UtilsCuda.h>
#include <Cuda/Common/VectorCuda.h>

#include <Core/Core.h>

#include <cuda_runtime.h>

namespace open3d {
/**
 * Client end
 */
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
    server_ = other.server();
    max_capacity_ = other.max_capacity_;
}

template<typename T>
ArrayCuda<T> &ArrayCuda<T>::operator=(const ArrayCuda<T> &other) {
    if (this != &other) {
        Release();

        server_ = other.server();
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
    assert(max_capacity_ < max_capacity);

    T *resized_data;
    CheckCuda(cudaMalloc(&(resized_data), sizeof(T) * max_capacity));

    int used_data_count = size();
    CheckCuda(cudaMemcpy(resized_data, server_->data_,
                         sizeof(T) * used_data_count,
                         cudaMemcpyDeviceToDevice));
    CheckCuda(cudaFree(server_->data_));
    server_->data_ = resized_data;

    max_capacity_ = max_capacity;
    server_->max_capacity_ = max_capacity;
}

template<typename T>
void ArrayCuda<T>::Create(int max_capacity) {
    assert(max_capacity > 0);

    if (server_ != nullptr) {
        PrintError("[ArrayCuda]: Already created, abort!\n");
        return;
    }

    server_ = std::make_shared<ArrayCudaServer<T >>();
    max_capacity_ = max_capacity;
    server_->max_capacity_ = max_capacity;

    CheckCuda(cudaMalloc(&(server_->data_), sizeof(T) * max_capacity));
    CheckCuda(cudaMemset(server_->data_, 0, sizeof(T) * max_capacity));
    CheckCuda(cudaMalloc(&(server_->iterator_), sizeof(int)));
    CheckCuda(cudaMemset(server_->iterator_, 0, sizeof(int)));
}

template<typename T>
void ArrayCuda<T>::Release() {
    if (server_ != nullptr && server_.use_count() == 1) {
        CheckCuda(cudaFree(server_->data_));
        CheckCuda(cudaFree(server_->iterator_));
    }

    server_ = nullptr;
    max_capacity_ = -1;
}

template<typename T>
void ArrayCuda<T>::CopyFromDeviceArray(const T *array, int size) {
    if (server_ == nullptr) {
        Create(size);
    } else if (server_->max_capacity_ < size) {
        PrintError("[ArrayCuda]: max capacity %d < %d, abort!\n",
                   max_capacity_, size);
        return;
    }
    CheckCuda(cudaMemcpy(server_->data_, array,
                         sizeof(T) * size,
                         cudaMemcpyDeviceToDevice));

}

template<typename T>
void ArrayCuda<T>::CopyTo(ArrayCuda<T> &other) const {
    assert(server_ != nullptr);

    if (this == &other) return;

    if (other.server_ == nullptr) {
        other.Create(max_capacity_);
    } else if (other.max_capacity_ < max_capacity_) {
        PrintError("[ArrayCuda]: other.max_capacity %d < %d, abort!\n",
                   max_capacity_, other.max_capacity_);
        return;
    }

    CheckCuda(cudaMemcpy(other.server()->data(), server_->data_,
                         sizeof(T) * max_capacity_,
                         cudaMemcpyDeviceToDevice));
}

template<typename T>
void ArrayCuda<T>::Upload(std::vector<T> &data) {
    assert(server_ != nullptr);

    int size = data.size();
    assert(size <= max_capacity_);
    CheckCuda(cudaMemcpy(server_->data_, data.data(),
                         sizeof(T) * size,
                         cudaMemcpyHostToDevice));
    CheckCuda(cudaMemcpy(server_->iterator_, &size,
                         sizeof(int),
                         cudaMemcpyHostToDevice));
}

template<typename T>
void ArrayCuda<T>::Upload(const T *data, int size) {
    assert(server_ != nullptr);

    assert(size <= max_capacity_);
    CheckCuda(cudaMemcpy(server_->data_, data,
                         sizeof(T) * size,
                         cudaMemcpyHostToDevice));
    CheckCuda(cudaMemcpy(server_->iterator_, &size,
                         sizeof(int),
                         cudaMemcpyHostToDevice));
}

template<class T>
std::vector<T> ArrayCuda<T>::Download() {
    assert(server_ != nullptr);

    std::vector<T> ret;
    int iterator_count = size();
    ret.resize(iterator_count);

    CheckCuda(cudaMemcpy(ret.data(), server_->data_,
                         sizeof(T) * iterator_count,
                         cudaMemcpyDeviceToHost));

    return ret;
}

template<typename T>
std::vector<T> ArrayCuda<T>::DownloadAll() {
    assert(server_ != nullptr);

    std::vector<T> ret;
    ret.resize(max_capacity_);

    CheckCuda(cudaMemcpy(ret.data(), server_->data_,
                         sizeof(T) * max_capacity_,
                         cudaMemcpyDeviceToHost));

    return ret; /* RVO will handle this (hopefully) */
}

template<typename T>
void ArrayCuda<T>::Fill(const T &val) {
    if (server_ == nullptr) return;
    ArrayCudaKernelCaller<T>::FillArrayKernelCaller(
        *server_, val, max_capacity_);
}

template<typename T>
void ArrayCuda<T>::Memset(int val) {
    assert(server_ != nullptr);

    CheckCuda(cudaMemset(server_->data_, val, sizeof(T) * max_capacity_));
}

template<class T>
void ArrayCuda<T>::Clear() {
    assert(server_ != nullptr);

    CheckCuda(cudaMemset(server_->iterator_, 0, sizeof(int)));
}

template<class T>
int ArrayCuda<T>::size() const {
    assert(server_ != nullptr);

    int ret;
    CheckCuda(cudaMemcpy(&ret, server_->iterator_,
                         sizeof(int),
                         cudaMemcpyDeviceToHost));
    return ret;
}

template<typename T>
void ArrayCuda<T>::set_size(int iterator_position) {
    assert(server_ != nullptr);

    assert(0 <= iterator_position && iterator_position <= max_capacity_);
    CheckCuda(cudaMemcpy(server_->iterator_, &iterator_position,
                         sizeof(int),
                         cudaMemcpyHostToDevice));
}
}