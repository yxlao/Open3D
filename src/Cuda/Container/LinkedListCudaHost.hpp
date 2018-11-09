/**
 * Created by wei on 18-4-2.
 */

#pragma once

#include "LinkedListCuda.h"
#include <Cuda/Common/UtilsCuda.h>

#include "MemoryHeapCuda.h"

#include <cassert>
#include <cstdio>

#include <Core/Core.h>


namespace open3d {
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
    server_ = other.server();
    memory_heap_ = other.memory_heap();
    max_capacity_ = other.max_capacity();
}

template<typename T>
LinkedListCuda<T> &LinkedListCuda<T>::operator=(
    const LinkedListCuda<T> &other) {
    if (this != &other) {
        Release();

        server_ = other.server();
        memory_heap_ = other.memory_heap();
        max_capacity_ = other.max_capacity();
    }

    return *this;
}

template<typename T>
void LinkedListCuda<T>::Create(int max_capacity,
                               MemoryHeapCuda<LinkedListNodeCuda<T>> &memory_heap) {
    assert(max_capacity > 0 && max_capacity < memory_heap.max_capacity_);
    if (server_ != nullptr) {
        PrintError("[LinkedListCuda] Already created, abort!\n");
        return;
    }

    server_ = std::make_shared<LinkedListCudaServer<T>>();
    max_capacity_ = max_capacity;
    memory_heap_ = memory_heap;

    CheckCuda(cudaMalloc(&server_->head_node_ptr_, sizeof(int)));
    CheckCuda(cudaMemset(server_->head_node_ptr_, NULLPTR_CUDA, sizeof(int)));

    CheckCuda(cudaMalloc(&server_->size_, sizeof(int)));
    CheckCuda(cudaMemset(server_->size_, 0, sizeof(int)));

    UpdateServer();
}

template<typename T>
void LinkedListCuda<T>::UpdateServer() {
    if (server_ != nullptr) {
        server_->max_capacity_ = max_capacity_;
        server_->memory_heap_ = *memory_heap_.server();
    }
}

template<typename T>
void LinkedListCuda<T>::Release() {
    if (server_ != nullptr && server_.use_count() == 1) {
        ClearLinkedListKernelCaller(*server_);
        CheckCuda(cudaFree(server_->head_node_ptr_));
        CheckCuda(cudaFree(server_->size_));
    }

    server_ = nullptr;
    max_capacity_ = -1;
}

template<typename T>
int LinkedListCuda<T>::size() {
    assert(server_ != nullptr);

    int ret;
    CheckCuda(cudaMemcpy(&ret, server_->size_,
                         sizeof(int),
                         cudaMemcpyDeviceToHost));
    return ret;
}

template<typename T>
void LinkedListCuda<T>::Insert(std::vector<int> &data) {
    assert(server_ != nullptr);

    ArrayCuda<T> data_cuda(data.size());
    data_cuda.Upload(data);
    InsertLinkedListKernelCaller(*server_, *data_cuda.server());
}

template<typename T>
void LinkedListCuda<T>::Find(std::vector<int> &query) {
    assert(server_ != nullptr);

    ArrayCuda<T> query_cuda(query.size());
    query_cuda.Upload(query);
    FindLinkedListKernelCaller(*server_, *query_cuda.server());
}

template<typename T>
void LinkedListCuda<T>::Delete(std::vector<int> &query) {
    assert(server_ != nullptr);

    ArrayCuda<T> query_cuda(query.size());
    query_cuda.Upload(query);
    DeleteLinkedListKernelCaller(*server_, *query_cuda.server());
}

template<typename T>
std::vector<T> LinkedListCuda<T>::Download() {
    assert(server_ != nullptr);

    int linked_list_size = size();
    if (linked_list_size == 0) return std::vector<T>();

    ArrayCuda<T> data(linked_list_size);
    DownloadLinkedListKernelCaller(*server_, *data.server());
    return data.DownloadAll();
}
};