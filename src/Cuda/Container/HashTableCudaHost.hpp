/*
 * Created by wei on 18-11-9.
 */

#pragma once

#include "HashTableCuda.h"

#include "ArrayCudaHost.hpp"
#include "LinkedListCudaHost.hpp"
#include "MemoryHeapCudaHost.hpp"

#include <Cuda/Common/UtilsCuda.h>

#include <cassert>
#include <tuple>
#include <vector>

#include <cuda_runtime.h>

#include <Core/Core.h>

namespace open3d {

namespace cuda {
/**
 * Client end
 */
template<typename Key, typename Value, typename Hasher>
HashTableCuda<Key, Value, Hasher>::HashTableCuda() {
    bucket_count_ = -1;
    max_value_capacity_ = -1;
    max_linked_list_node_capacity_ = -1;
}

template<typename Key, typename Value, typename Hasher>
HashTableCuda<Key, Value, Hasher>::~HashTableCuda() {
    Release();
}

template<typename Key, typename Value, typename Hasher>
HashTableCuda<Key, Value, Hasher>::HashTableCuda(
    const HashTableCuda<Key, Value, Hasher> &other) {

    bucket_count_ = other.bucket_count_;
    max_value_capacity_ = other.max_value_capacity_;
    max_linked_list_node_capacity_ = other.max_linked_list_node_capacity_;
    hasher_ = other.hasher();

    server_ = other.server();

    /** No need to call UpdateServer(), they should have been copied to
     * other.server() and assigned to server_; */

    memory_heap_entry_list_node_ = other.memory_heap_entry_list_node();
    memory_heap_value_ = other.memory_heap_value();

    entry_array_ = other.entry_array();
    entry_list_array_ = other.entry_list_array();
    lock_array_ = other.lock_array();
    assigned_entry_array_ = other.assigned_entry_array();
}

template<typename Key, typename Value, typename Hasher>
HashTableCuda<Key, Value, Hasher> &HashTableCuda<Key, Value, Hasher>::operator=(
    const HashTableCuda<Key, Value, Hasher> &other) {
    if (this != &other) {
        bucket_count_ = other.bucket_count_;
        max_value_capacity_ = other.max_value_capacity_;
        max_linked_list_node_capacity_ = other.max_linked_list_node_capacity_;
        hasher_ = other.hasher();

        server_ = other.server();

        memory_heap_entry_list_node_ = other.memory_heap_entry_list_node();
        memory_heap_value_ = other.memory_heap_value();

        entry_array_ = other.entry_array();
        entry_list_array_ = other.entry_list_array();
        lock_array_ = other.lock_array();
        assigned_entry_array_ = other.assigned_entry_array();
    }

    return *this;
}

template<typename Key, typename Value, typename Hasher>
void HashTableCuda<Key, Value, Hasher>::Create(
    int bucket_count, int value_capacity) {
    assert(bucket_count > 0 && value_capacity > 0);

    if (server_ != nullptr) {
        PrintError("[HashTableCuda] Already created, abort!\n");
        return;
    }

    server_ = std::make_shared<HashTableCudaServer<Key, Value, Hasher>>();

    bucket_count_ = bucket_count; /* (* BUCKET_SIZE), 2D array */
    hasher_ = Hasher(bucket_count);

    max_value_capacity_ = value_capacity;
    max_linked_list_node_capacity_ = bucket_count * BUCKET_SIZE;

    memory_heap_entry_list_node_.Create(max_linked_list_node_capacity_);
    memory_heap_value_.Create(max_value_capacity_);

    entry_array_.Create(bucket_count * BUCKET_SIZE);
    entry_list_array_.Create(bucket_count);
    lock_array_.Create(bucket_count);
    assigned_entry_array_.Create(bucket_count * BUCKET_SIZE +
        max_linked_list_node_capacity_);

    /** We have to manually allocate these raw ptrs for the linkedlists that
     * are initialized on CUDA (they don't have corresponding clients, so let
     * HashTable class be its client)
     */
    CheckCuda(cudaMalloc(&server_->entry_list_head_node_ptrs_memory_pool_,
                         sizeof(int) * bucket_count));
    CheckCuda(cudaMalloc(&server_->entry_list_size_ptrs_memory_pool_,
                         sizeof(int) * bucket_count));
    UpdateServer();

    HashTableCudaKernelCaller<Key, Value, Hasher>::
    CreateHashTableEntriesKernelCaller(*server_, bucket_count_);
}

template<typename Key, typename Value, typename Hasher>
void HashTableCuda<Key, Value, Hasher>::Release() {
    /** Since we Release, we don't care about the content of the linked list
     * array. They are stored in the memory_heap and will be Releaseed anyway.
     */
    if (server_ != nullptr && server_.use_count() == 1) {
        HashTableCudaKernelCaller<Key, Value, Hasher>::
        ReleaseHashTableEntriesKernelCaller(*server_, bucket_count_);
        entry_array_.Release();
        entry_list_array_.Release();
        lock_array_.Release();
        assigned_entry_array_.Release();

        memory_heap_entry_list_node_.Release();
        memory_heap_value_.Release();
        CheckCuda(cudaFree(server_->entry_list_head_node_ptrs_memory_pool_));
        CheckCuda(cudaFree(server_->entry_list_size_ptrs_memory_pool_));
    }

    server_ = nullptr;
    bucket_count_ = -1;
    max_value_capacity_ = -1;
    max_linked_list_node_capacity_ = -1;
}

template<typename Key, typename Value, typename Hasher>
void HashTableCuda<Key, Value, Hasher>::UpdateServer() {
    if (server_ != nullptr) {
        server_->hasher_ = hasher_;
        server_->bucket_count_ = bucket_count_;

        server_->memory_heap_entry_list_node_ =
            *memory_heap_entry_list_node_.server();
        server_->memory_heap_value_ = *memory_heap_value_.server();
        server_->entry_array_ = *entry_array_.server();
        server_->entry_list_array_ = *entry_list_array_.server();
        server_->lock_array_ = *lock_array_.server();
        server_->assigned_entry_array_ = *assigned_entry_array_.server();
    }
}

template<typename Key, typename Value, typename Hasher>
void HashTableCuda<Key, Value, Hasher>::Reset() {
    assert(server_ != nullptr);

    memory_heap_entry_list_node_.Reset();
    memory_heap_value_.Reset();
    ResetEntries();
    ResetLocks();
}

template<typename Key, typename Value, typename Hasher>
void HashTableCuda<Key, Value, Hasher>::ResetEntries() {
    assert(server_ != nullptr);

    HashTableCudaKernelCaller<Key, Value, Hasher>::
    ResetHashTableEntriesKernelCaller(*server_, bucket_count_);
}

template<typename Key, typename Value, typename Hasher>
void HashTableCuda<Key, Value, Hasher>::ResetLocks() {
    assert(server_ != nullptr);

    lock_array_.Memset(0);
}

template<typename Key, typename Value, typename Hasher>
void HashTableCuda<Key, Value, Hasher>::GetAssignedEntries() {
    assert(server_ != nullptr);

    /* Reset counter */
    assigned_entry_array_.Clear();

    HashTableCudaKernelCaller<Key, Value, Hasher>::
    GetHashTableAssignedEntriesKernelCaller(*server_, bucket_count_);
}

template<typename Key, typename Value, typename Hasher>
void HashTableCuda<Key, Value, Hasher>::New(
    std::vector<Key> &keys, std::vector<Value> &values) {
    assert(server_ != nullptr);

    ArrayCuda<Key> keys_cuda(keys.size());
    keys_cuda.Upload(keys);
    ArrayCuda<Value> values_cuda(values.size());
    values_cuda.Upload(values);

    HashTableCudaKernelCaller<Key, Value, Hasher>::
    InsertHashTableEntriesKernelCaller(*server_,
                                       *keys_cuda.server(),
                                       *values_cuda.server(),
                                       keys.size(),
                                       bucket_count_);
}

template<typename Key, typename Value, typename Hasher>
void HashTableCuda<Key, Value, Hasher>
::Delete(std::vector<Key> &keys) {
    assert(server_ != nullptr);

    ArrayCuda<Key> keys_cuda(keys.size());
    keys_cuda.Upload(keys);

    HashTableCudaKernelCaller<Key, Value, Hasher>::
    DeleteHashTableEntriesKernelCaller(*server_,
                                       *keys_cuda.server(), keys.size(),
                                       bucket_count_);
}

template<typename Key, typename Value, typename Hasher>
std::tuple<std::vector<Key>, std::vector<Value>>
HashTableCuda<Key, Value, Hasher>::Download() {
    assert(server_ != nullptr);

    std::vector<Key> keys;
    std::vector<Value> values;

    GetAssignedEntries();

    int assigned_entry_array_size = assigned_entry_array_.size();
    if (assigned_entry_array_size == 0)
        return std::make_tuple(keys, values);

    GetAssignedEntries();
    std::vector<Entry> assigned_entries = assigned_entry_array_.Download();
    std::vector<Value> memory_heap_values = memory_heap_value_.DownloadValue();

    /* It could be very memory-consuming when we try to dump VoxelBlocks ... */
    keys.resize(assigned_entry_array_size);
    values.resize(assigned_entry_array_size);
    for (int i = 0; i < assigned_entry_array_size; ++i) {
        Entry &entry = assigned_entries[i];
        keys[i] = entry.key;
        values[i] = memory_heap_values[entry.internal_addr];
    }

    return std::make_tuple(keys, values);
}

template<typename Key, typename Value, typename Hasher>
std::vector<HashEntry<Key>> HashTableCuda<Key, Value, Hasher>
::DownloadAssignedEntries() {
    assert(server_ != nullptr);

    std::vector<Entry> ret;
    int assigned_entry_array_size = assigned_entry_array_.size();
    if (assigned_entry_array_size == 0) return ret;
    return assigned_entry_array_.Download();
}

template<typename Key, typename Value, typename Hasher>
std::tuple<std::vector<int>, std::vector<int>>
HashTableCuda<Key, Value, Hasher>::Profile() {
    assert(server_ != nullptr);

    ArrayCuda<int> array_entry_count_cuda(bucket_count_);
    ArrayCuda<int> list_entry_count_cuda(bucket_count_);

    HashTableCudaKernelCaller<Key, Value, Hasher>::
    ProfileHashTableKernelCaller(*server_,
                                 *array_entry_count_cuda.server(),
                                 *list_entry_count_cuda.server(),
                                 bucket_count_);

    std::vector<int> array_entry_count = array_entry_count_cuda.DownloadAll();
    std::vector<int> list_entry_count = list_entry_count_cuda.DownloadAll();

    return std::make_tuple(array_entry_count, list_entry_count);
}
} // cuda
} // open3d