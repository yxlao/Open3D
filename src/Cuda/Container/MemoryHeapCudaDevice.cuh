//
// Created by wei on 18-3-29.
//

#pragma once

#include "MemoryHeapCuda.h"
#include <Cuda/Common/UtilsCuda.h>

#include <Core/Core.h>

#include <cassert>

namespace open3d {

/**
 * Server end
 * In fact we don't know which indices are used and which are freed,
 * we can only give a coarse boundary test.
 */
template<typename T>
__device__
int &MemoryHeapCudaServer<T>::internal_addr_at(size_t index) {
#ifdef CUDA_DEBUG_ENABLE_ASSERTION
    assert(index < max_capacity_);
#endif
    return heap_[index];
}

template<typename T>
__device__
T &MemoryHeapCudaServer<T>::value_at(size_t addr) {
#ifdef CUDA_DEBUG_ENABLE_ASSERTION
    assert(addr < max_capacity_);
#endif
    return data_[addr];
}

template<typename T>
__device__
const T &MemoryHeapCudaServer<T>::value_at(size_t addr) const {
#ifdef CUDA_DEBUG_ENABLE_ASSERTION
    assert(addr < max_capacity_);
#endif
    return data_[addr];
}


/**
 * The @value array is FIXED.
 * The @heap array stores the addresses of the values.
 * Only the unallocated part is maintained.
 * (ONLY care about the heap above the heap counter. Below is meaningless.)
 * ---------------------------------------------------------------------
 * heap  ---Malloc-->  heap  ---Malloc-->  heap  ---Free(0)-->  heap
 * N-1                 N-1                  N-1                  N-1   |
 *  .                   .                    .                    .    |
 *  .                   .                    .                    .    |
 *  .                   .                    .                    .    |
 *  3                   3                    3                    3    |
 *  2                   2                    2 <-                 2    |
 *  1                   1 <-                 1                    0 <- |
 *  0 <- heap_counter   0                    0                    0
 */
template<class T>
__device__
int MemoryHeapCudaServer<T>::Malloc() {
    int index = atomicAdd(heap_counter_, 1);
#ifdef CUDA_DEBUG_ENABLE_ASSERTION
    assert(index < max_capacity_);
#endif
    return heap_[index];
}

template<class T>
__device__
void MemoryHeapCudaServer<T>::Free(size_t addr) {
    int index = atomicSub(heap_counter_, 1);
#ifdef CUDA_DEBUG_ENABLE_ASSERTION
    assert(index >= 1);
#endif
    heap_[index - 1] = addr;
}

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