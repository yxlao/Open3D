//
// Created by wei on 18-11-9.
//

#pragma once

#include "MemoryHeapCuda.h"
#include <Cuda/Common/UtilsCuda.h>

#include <Core/Core.h>
#include <cassert>

namespace open3d {
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
    server_ = other.server();
    max_capacity_ = other.max_capacity_;
}

template<typename T>
MemoryHeapCuda<T>& MemoryHeapCuda<T>::operator=(
    const MemoryHeapCuda<T> &other) {
    if (this != &other) {
        Release();

        server_ = other.server();
        max_capacity_ = other.max_capacity_;
    }
    return *this;
}

template<typename T>
void MemoryHeapCuda<T>::Resize(int new_max_capacity) {
    assert(max_capacity_ < new_max_capacity);

    auto new_server = std::make_shared<MemoryHeapCudaServer<T>>();
    new_server->max_capacity_ = new_max_capacity;
    CheckCuda(cudaMalloc(&new_server->heap_counter_, sizeof(int)));
    CheckCuda(cudaMalloc(&new_server->heap_, sizeof(int) * new_max_capacity));
    CheckCuda(cudaMalloc(&new_server->data_, sizeof(T) * new_max_capacity));

    MemoryHeapCudaKernelCaller<T>::ResizeMemoryHeapKernelCaller(
        *server_, *new_server, new_max_capacity);

    CheckCuda(cudaFree(server_->heap_counter_));
    CheckCuda(cudaFree(server_->heap_));
    CheckCuda(cudaFree(server_->data_));

    server_->heap_counter_ = new_server->heap_counter();
    server_->heap_ = new_server->heap();
    server_->data_ = new_server->data();
    server_->max_capacity_ = new_max_capacity;

    max_capacity_ = new_max_capacity;
}

template<typename T>
void MemoryHeapCuda<T>::Create(int max_capacity) {
    assert(max_capacity > 0);

    if (server_ != nullptr) {
        PrintError("[MemoryHeapCuda] Already created, abort!\n");
        return;
    }

    server_ = std::make_shared<MemoryHeapCudaServer<T>>();
    max_capacity_ = max_capacity;
    server_->max_capacity_ = max_capacity;

    CheckCuda(cudaMalloc(&(server_->heap_counter_), sizeof(int)));
    CheckCuda(cudaMalloc(&(server_->heap_), sizeof(int) * max_capacity));
    CheckCuda(cudaMalloc(&(server_->data_), sizeof(T) * max_capacity));

    Reset();
}

template<typename T>
void MemoryHeapCuda<T>::Release() {
    if (server_ != nullptr && server_.use_count() == 1) {
        CheckCuda(cudaFree(server_->heap_counter_));
        CheckCuda(cudaFree(server_->heap_));
        CheckCuda(cudaFree(server_->data_));
    }

    server_ = nullptr;
    max_capacity_ = -1;
}

template<typename T>
void MemoryHeapCuda<T>::Reset() {
    assert(server_ != nullptr);

    MemoryHeapCudaKernelCaller<T>::
    ResetMemoryHeapKernelCaller(*server_, max_capacity_);
    CheckCuda(cudaDeviceSynchronize());
    CheckCuda(cudaGetLastError());

    int heap_counter = 0;
    CheckCuda(cudaMemcpy(server_->heap_counter_, &heap_counter,
                         sizeof(int), cudaMemcpyHostToDevice));
}

template<typename T>
std::vector<int> MemoryHeapCuda<T>::DownloadHeap() {
    assert(server_ != nullptr);

    std::vector<int> ret;
    ret.resize(max_capacity_);
    CheckCuda(cudaMemcpy(ret.data(), server_->heap_,
                         sizeof(int) * max_capacity_,
                         cudaMemcpyDeviceToHost));
    return ret;
}

template<typename T>
std::vector<T> MemoryHeapCuda<T>::DownloadValue() {
    assert(server_ != nullptr);

    std::vector<T> ret;
    ret.resize(max_capacity_);
    CheckCuda(cudaMemcpy(ret.data(), server_->data_,
                         sizeof(T) * max_capacity_,
                         cudaMemcpyDeviceToHost));
    return ret;
}

template<typename T>
int MemoryHeapCuda<T>::HeapCounter() {
    assert(server_ != nullptr);

    int heap_counter;
    CheckCuda(cudaMemcpy(&heap_counter, server_->heap_counter_,
                         sizeof(int),
                         cudaMemcpyDeviceToHost));
    return heap_counter;
}
};