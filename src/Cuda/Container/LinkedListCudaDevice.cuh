/**
 * Created by wei on 18-4-2.
 */

#pragma once

#include "LinkedListCuda.h"
#include <Cuda/Common/UtilsCuda.h>

#include "MemoryHeapCudaDevice.cuh"

#include <cassert>
#include <cstdio>


namespace open3d {
/**
 * Server end
 */
template<typename T>
__device__
void LinkedListCudaServer<T>::Clear() {
    int node_ptr = *head_node_ptr_;
    while (node_ptr != NULLPTR_CUDA) {
        int next_node_ptr = memory_heap_.value_at(node_ptr).next_node_ptr;
        memory_heap_.Free(node_ptr);
        node_ptr = next_node_ptr;
        (*size_)--;
    }
    *head_node_ptr_ = NULLPTR_CUDA;

#ifdef CUDA_DEBUG_ENABLE_ASSERTION
    assert((*size_) == 0);
#endif
}

template<typename T>
__device__
void LinkedListCudaServer<T>::Insert(T value) {
    int node_ptr = memory_heap_.Malloc();

#ifdef CUDA_DEBUG_ENABLE_ASSERTION
    assert(*size_ < max_capacity_);
#endif

    memory_heap_.value_at(node_ptr).data = value;
    memory_heap_.value_at(node_ptr).next_node_ptr = *head_node_ptr_;
    *head_node_ptr_ = node_ptr;

    (*size_)++;
}

/**
 * Cuda version of Createialization.
 * It is more intuitive to directly allocate it in kernel with malloc(),
 * but it fails with not too many number of operations
 * @TODO check why.
 * A way around is to pre-allocate it on the host side using CheckCuda(cudaMalloc(),
 * and assign these space directly to the pointers.
 */
template<typename T>
__device__
void LinkedListCudaServer<T>::Create(
    LinkedListCudaServer<T>::MemoryHeapServer &memory_heap_server,
    int *head_node_ptr,
    int *size_ptr) {
    max_capacity_ = memory_heap_server.max_capacity_;
    memory_heap_ = memory_heap_server;
    head_node_ptr_ = head_node_ptr;
    size_ = size_ptr;
    (*head_node_ptr_) = NULLPTR_CUDA;
    (*size_) = 0;
}

/* Release the assigned pointers externally */
template<typename T>
__device__
void LinkedListCudaServer<T>::Release() {}

template<typename T>
__device__
int LinkedListCudaServer<T>::Delete(const int node_ptr) {
    if (*head_node_ptr_ == NULLPTR_CUDA || node_ptr == NULLPTR_CUDA) {
#ifdef CUDA_DEBUG_ENABLE_PRINTF
        printf("Error: Invalid pointer or linked list!\n");
#endif
        return ContainerReturnCode::LinkedListEntryNotFound;
    }

    /* 1. Head */
    if (*head_node_ptr_ == node_ptr) {
        *head_node_ptr_ = memory_heap_.value_at(*head_node_ptr_).next_node_ptr;
        memory_heap_.Free(node_ptr);
        (*size_)--;
        return ContainerReturnCode::Success;
    }

    /* 2. Search in the linked list for its predecessor */
    int node_ptr_pred = *head_node_ptr_;
    while (memory_heap_.value_at(node_ptr_pred).next_node_ptr != node_ptr
        && memory_heap_.value_at(node_ptr_pred).next_node_ptr != NULLPTR_CUDA) {
        node_ptr_pred = memory_heap_.value_at(node_ptr_pred).next_node_ptr;
    }

    if (memory_heap_.value_at(node_ptr_pred).next_node_ptr == NULLPTR_CUDA) {
#ifdef CUDA_DEBUG_ENABLE_PRINTF
        printf("Error: Node_ptr %d not found!\n", node_ptr);
#endif
        return ContainerReturnCode::LinkedListEntryNotFound;
    }

    memory_heap_.value_at(node_ptr_pred).next_node_ptr =
        memory_heap_.value_at(node_ptr).next_node_ptr;
    memory_heap_.Free(node_ptr);
    (*size_)--;
    return ContainerReturnCode::Success;
}

template<typename T>
__device__
int LinkedListCudaServer<T>::Find(T value) const {
    int node_ptr = *head_node_ptr_;
    while (node_ptr != NULLPTR_CUDA) {
        if (memory_heap_.value_at(node_ptr).data == value)
            return node_ptr;
        node_ptr = memory_heap_.value_at(node_ptr).next_node_ptr;
    }

#ifdef CUDA_DEBUG_ENABLE_PRINTF
    printf("Error: Value not found!\n");
#endif
    return NULLPTR_CUDA;
}

template<class T>
__device__
int LinkedListCudaServer<T>::FindAndDelete(T value) {
    if (*head_node_ptr_ == NULLPTR_CUDA) {
#ifdef CUDA_DEBUG_ENABLE_PRINTF
        printf("Empty linked list!\n");
#endif
        return ContainerReturnCode::LinkedListEntryNotFound;
    }

    /* 1. Head */
    if (memory_heap_.value_at(*head_node_ptr_).data == value) {
        /* NULL_PTR and ! NULL_PTR, both cases are ok */
        int next_node_ptr = memory_heap_.value_at(*head_node_ptr_)
            .next_node_ptr;
        memory_heap_.Free(*head_node_ptr_);
        *head_node_ptr_ = next_node_ptr;
        (*size_)--;
        return ContainerReturnCode::Success;
    }

    /* 2. Search in the linked list for its predecessor */
    int node_ptr_pred = *head_node_ptr_;
    int node_ptr_curr = memory_heap_.value_at(*head_node_ptr_).next_node_ptr;
    while (node_ptr_curr != NULLPTR_CUDA) {
        T &data = memory_heap_.value_at(node_ptr_curr).data;
        if (data == value) break;
        node_ptr_pred = node_ptr_curr;
        node_ptr_curr = memory_heap_.value_at(node_ptr_curr).next_node_ptr;
    }

    if (node_ptr_curr == NULLPTR_CUDA) {
#ifdef CUDA_DEBUG_ENABLE_PRINTF
        printf("Error: Value not found!\n");
#endif
        return ContainerReturnCode::LinkedListEntryNotFound;
    }

    memory_heap_.value_at(node_ptr_pred).next_node_ptr =
        memory_heap_.value_at(node_ptr_curr).next_node_ptr;
    memory_heap_.Free(node_ptr_curr);
    (*size_)--;
    return ContainerReturnCode::Success;
}
};