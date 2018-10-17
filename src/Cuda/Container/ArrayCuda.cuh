/**
 * Created by wei on 18-4-2.
 */

#pragma once

#include "ArrayCuda.h"
#include <Cuda/Common/UtilsCuda.h>

#include <cuda_runtime.h>
#include <cassert>
#include <Core/Core.h>

namespace open3d {

/**
 * Server end
 */
template<typename T>
__device__
int ArrayCudaServer<T>::push_back(T value) {
    int addr = atomicAdd(iterator_, 1);
    data_[addr] = value;
    return addr;
}

template<typename T>
__device__
T &ArrayCudaServer<T>::get(size_t index) {
    return data_[index];
}

template<typename T>
__device__
T &ArrayCudaServer<T>::operator[](size_t index) {
    return data_[index];
}


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
    max_capacity_ = other.max_capacity();
}

template<typename T>
ArrayCuda<T> &ArrayCuda<T>::operator=(const ArrayCuda<T> &other) {
    if (this != &other) {
        Release();

        server_ = other.server();
        max_capacity_ = other.max_capacity();
    }
    return *this;
}

template<typename T>
ArrayCuda<T>::~ArrayCuda() {
    Release();
}

template<typename T>
void ArrayCuda<T>::Create(int max_capacity) {
    assert(max_capacity > 0);
    if (server_ != nullptr) {
        PrintError("Already created, stop re-creating!\n");
        return;
    }

    max_capacity_ = max_capacity;
    server_ = std::make_shared<ArrayCudaServer<T>>();
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
void ArrayCuda<T>::FromCudaArray(const T *array, int size) {
    if (server_ == nullptr) {
        Create(size);
    } else {
        if (server_->max_capacity_ < size) {
            PrintError("Array capacity too small, abort!\n");
            return;
        }
    }
    CheckCuda(cudaMemcpy(server_->data_, array,
                         sizeof(T) * size,
                         cudaMemcpyDeviceToDevice));

}

template<typename T>
void ArrayCuda<T>::CopyTo(ArrayCuda<T> &other) const {
    if (this == &other) return;

    if (other.server_ == nullptr) {
        other.Create(max_capacity_);
    } else if (other.max_capacity() != max_capacity_) {
        PrintError("Incompatible array size!\n");
        return;
    }

    CheckCuda(cudaMemcpy(other.server()->data(), server_->data_,
                         sizeof(T) * max_capacity_,
                         cudaMemcpyDeviceToDevice));
}

template<typename T>
void ArrayCuda<T>::Upload(std::vector<T> &data) {
    int size = data.size();
    assert(size < max_capacity_);
    CheckCuda(cudaMemcpy(server_->data_, data.data(), sizeof(T) * size,
                         cudaMemcpyHostToDevice));
    CheckCuda(cudaMemcpy(server_->iterator_, &size,
                         sizeof(int),
                         cudaMemcpyHostToDevice));
}

template<typename T>
void ArrayCuda<T>::Upload(const T *data, int size) {
    assert(size < max_capacity_);
    CheckCuda(cudaMemcpy(server_->data_, data, sizeof(T) * size,
                         cudaMemcpyHostToDevice));
    CheckCuda(cudaMemcpy(server_->iterator_, &size,
                         sizeof(int),
                         cudaMemcpyHostToDevice));
}

template<class T>
std::vector<T> ArrayCuda<T>::Download() {
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
    std::vector<T> ret;
    ret.resize(max_capacity_);

    CheckCuda(cudaMemcpy(ret.data(), server_->data_,
                         sizeof(T) * max_capacity_,
                         cudaMemcpyDeviceToHost));

    return ret; /* RVO will handle this (hopefully) */
}

template<typename T>
void ArrayCuda<T>::Fill(const T &val) {
    const int threads = THREAD_1D_UNIT;
    const int blocks = UPPER_ALIGN(max_capacity_, THREAD_1D_UNIT);
    FillArrayKernel << < blocks, threads >> > (*server_, val);
    CheckCuda(cudaDeviceSynchronize());
}

template<typename T>
void ArrayCuda<T>::Memset(int val) {
    CheckCuda(cudaMemset(server_->data_, val, sizeof(T) * max_capacity_));
}

template<class T>
void ArrayCuda<T>::Clear() {
    CheckCuda(cudaMemset(server_->iterator_, 0, sizeof(int)));
}

template<class T>
int ArrayCuda<T>::size() {
    int ret;
    CheckCuda(cudaMemcpy(&ret, server_->iterator_,
                         sizeof(int),
                         cudaMemcpyDeviceToHost));
    return ret;
}

template<typename T>
void ArrayCuda<T>::set_size(int iterator_position) {
    assert(0 <= iterator_position && iterator_position <= max_capacity_);
    CheckCuda(cudaMemcpy(server_->iterator_, &iterator_position,
                         sizeof(int),
                         cudaMemcpyHostToDevice));
}
}