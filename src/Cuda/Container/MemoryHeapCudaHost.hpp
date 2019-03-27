//
// Created by wei on 18-11-9.
//

#pragma once

#include "MemoryHeapCuda.h"
#include <Cuda/Common/UtilsCuda.h>

#include <cassert>

namespace open3d {

namespace cuda {
/**
 * Client end
 */
template<typename T>
MemoryHeapCuda<T>::MemoryHeapCuda() {
    max_capacity_ = -1;
}

template<typename T>
MemoryHeapCuda<T>::~MemoryHeapCuda() {
    Release();
}

template<typename T>
MemoryHeapCuda<T>::MemoryHeapCuda(const MemoryHeapCuda<T> &other) {
    device_ = other.device_;
    max_capacity_ = other.max_capacity_;
}

template<typename T>
MemoryHeapCuda<T> &MemoryHeapCuda<T>::operator=(
    const MemoryHeapCuda<T> &other) {
    if (this != &other) {
        Release();

        device_ = other.device_;
        max_capacity_ = other.max_capacity_;
    }
    return *this;
}

template<typename T>
void MemoryHeapCuda<T>::Resize(int new_max_capacity) {
    assert(max_capacity_ < new_max_capacity);

    MemoryHeapCuda new_memory_heap;
    new_memory_heap.Create(new_max_capacity);

    MemoryHeapCudaKernelCaller<T>::Resize(*this, new_memory_heap);
    *this = new_memory_heap;
}

template<typename T>
void MemoryHeapCuda<T>::Create(int max_capacity) {
    assert(max_capacity > 0);

    if (device_ != nullptr) {
        utility::PrintError("[MemoryHeapCuda] Already created, abort!\n");
        return;
    }

    device_ = std::make_shared<MemoryHeapCudaDevice<T>>();
    max_capacity_ = max_capacity;
    device_->max_capacity_ = max_capacity;

    CheckCuda(cudaMalloc(&(device_->heap_counter_), sizeof(int)));
    CheckCuda(cudaMalloc(&(device_->heap_), sizeof(int) * max_capacity));
    CheckCuda(cudaMalloc(&(device_->data_), sizeof(T) * max_capacity));

    Reset();
}

template<typename T>
void MemoryHeapCuda<T>::Release() {
    if (device_ != nullptr && device_.use_count() == 1) {
        CheckCuda(cudaFree(device_->heap_counter_));
        CheckCuda(cudaFree(device_->heap_));
        CheckCuda(cudaFree(device_->data_));
    }

    device_ = nullptr;
    max_capacity_ = -1;
}

template<typename T>
void MemoryHeapCuda<T>::Reset() {
    assert(device_ != nullptr);

    MemoryHeapCudaKernelCaller<T>::Reset(*this);
    CheckCuda(cudaDeviceSynchronize());
    CheckCuda(cudaGetLastError());

    int heap_counter = 0;
    CheckCuda(cudaMemcpy(device_->heap_counter_, &heap_counter,
                         sizeof(int), cudaMemcpyHostToDevice));
}

template<typename T>
std::vector<int> MemoryHeapCuda<T>::DownloadHeap() {
    assert(device_ != nullptr);

    std::vector<int> ret;
    ret.resize(max_capacity_);
    CheckCuda(cudaMemcpy(ret.data(), device_->heap_,
                         sizeof(int) * max_capacity_,
                         cudaMemcpyDeviceToHost));
    return ret;
}

template<typename T>
std::vector<T> MemoryHeapCuda<T>::DownloadValue() {
    assert(device_ != nullptr);

    std::vector<T> ret;
    ret.resize(max_capacity_);
    CheckCuda(cudaMemcpy(ret.data(), device_->data_,
                         sizeof(T) * max_capacity_,
                         cudaMemcpyDeviceToHost));
    return ret;
}

template<typename T>
int MemoryHeapCuda<T>::HeapCounter() {
    assert(device_ != nullptr);

    int heap_counter;
    CheckCuda(cudaMemcpy(&heap_counter, device_->heap_counter_,
                         sizeof(int),
                         cudaMemcpyDeviceToHost));
    return heap_counter;
}
} // cuda
} // open3d