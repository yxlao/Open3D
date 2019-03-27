/**
 * Created by wei on 18-4-2.
 */

#pragma once

#include "LinkedListCuda.h"
#include <Cuda/Common/UtilsCuda.h>

#include "MemoryHeapCuda.h"

#include <cassert>
#include <cstdio>

namespace open3d {

namespace cuda {
/**
 * Client end
 */
template<typename T>
LinkedListCuda<T>::LinkedListCuda() {
    max_capacity_ = -1;
}

template<typename T>
LinkedListCuda<T>::~LinkedListCuda() {
    Release();
}

template<typename T>
LinkedListCuda<T>::LinkedListCuda(const LinkedListCuda<T> &other) {
    device_ = other.device_;
    memory_heap_ = other.memory_heap();
    max_capacity_ = other.max_capacity_;
}

template<typename T>
LinkedListCuda<T> &LinkedListCuda<T>::operator=(
    const LinkedListCuda<T> &other) {
    if (this != &other) {
        Release();

        device_ = other.device_;
        memory_heap_ = other.memory_heap();
        max_capacity_ = other.max_capacity_;
    }

    return *this;
}

template<typename T>
void LinkedListCuda<T>::Create(int max_capacity,
                               MemoryHeapCuda<LinkedListNodeCuda<T>> &memory_heap) {
    assert(max_capacity > 0 && max_capacity < memory_heap.max_capacity_);
    if (device_ != nullptr) {
        utility::PrintError("[LinkedListCuda] Already created, abort!\n");
        return;
    }

    device_ = std::make_shared<LinkedListCudaDevice<T>>();
    max_capacity_ = max_capacity;
    memory_heap_ = memory_heap;

    CheckCuda(cudaMalloc(&device_->head_node_ptr_, sizeof(int)));
    CheckCuda(cudaMemset(device_->head_node_ptr_, NULLPTR_CUDA, sizeof(int)));

    CheckCuda(cudaMalloc(&device_->size_, sizeof(int)));
    CheckCuda(cudaMemset(device_->size_, 0, sizeof(int)));

    UpdateDevice();
}

template<typename T>
void LinkedListCuda<T>::UpdateDevice() {
    if (device_ != nullptr) {
        device_->max_capacity_ = max_capacity_;
        device_->memory_heap_ = *memory_heap_.device_;
    }
}

template<typename T>
void LinkedListCuda<T>::Release() {
    if (device_ != nullptr && device_.use_count() == 1) {
        LinkedListCudaKernelCaller<T>::Clear(*this);
        CheckCuda(cudaFree(device_->head_node_ptr_));
        CheckCuda(cudaFree(device_->size_));
    }

    device_ = nullptr;
    max_capacity_ = -1;
}

template<typename T>
int LinkedListCuda<T>::size() {
    assert(device_ != nullptr);

    int ret;
    CheckCuda(cudaMemcpy(&ret, device_->size_,
                         sizeof(int),
                         cudaMemcpyDeviceToHost));
    return ret;
}

template<typename T>
void LinkedListCuda<T>::Insert(std::vector<int> &data) {
    assert(device_ != nullptr);

    ArrayCuda<T> data_cuda(data.size());
    data_cuda.Upload(data);
    LinkedListCudaKernelCaller<T>::Insert(*this, data_cuda);
}

template<typename T>
void LinkedListCuda<T>::Find(std::vector<int> &query) {
    assert(device_ != nullptr);

    ArrayCuda<T> query_cuda(query.size());
    query_cuda.Upload(query);
    LinkedListCudaKernelCaller<T>::Find(*this, query_cuda);
}

template<typename T>
void LinkedListCuda<T>::Delete(std::vector<int> &query) {
    assert(device_ != nullptr);

    ArrayCuda<T> query_cuda(query.size());
    query_cuda.Upload(query);
    LinkedListCudaKernelCaller<T>::Delete(*this, query_cuda);
}

template<typename T>
std::vector<T> LinkedListCuda<T>::Download() {
    assert(device_ != nullptr);

    int linked_list_size = size();
    if (linked_list_size == 0) return std::vector<T>();

    ArrayCuda<T> data(linked_list_size);
    LinkedListCudaKernelCaller<T>::Download(*this, data);
    return data.DownloadAll();
}
} // cuda
} // open3d