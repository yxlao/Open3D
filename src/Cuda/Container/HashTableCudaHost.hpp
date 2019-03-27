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

    device_ = other.device_;

    /** No need to call UpdateDevice(), they should have been copied to
     * other.device_ and assigned to device_; */

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

        device_ = other.device_;

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

    if (device_ != nullptr) {
        utility::PrintError("[HashTableCuda] Already created, abort!\n");
        return;
    }

    device_ = std::make_shared<HashTableCudaDevice<Key, Value, Hasher>>();

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
    CheckCuda(cudaMalloc(&device_->entry_list_head_node_ptrs_memory_pool_,
                         sizeof(int) * bucket_count));
    CheckCuda(cudaMalloc(&device_->entry_list_size_ptrs_memory_pool_,
                         sizeof(int) * bucket_count));
    UpdateDevice();

    HashTableCudaKernelCaller<Key, Value, Hasher>::Create(*this);
}

template<typename Key, typename Value, typename Hasher>
void HashTableCuda<Key, Value, Hasher>::Release() {
    /** Since we Release, we don't care about the content of the linked list
     * array. They are stored in the memory_heap and will be Releaseed anyway.
     */
    if (device_ != nullptr && device_.use_count() == 1) {
        HashTableCudaKernelCaller<Key, Value, Hasher>::Release(*this);
        entry_array_.Release();
        entry_list_array_.Release();
        lock_array_.Release();
        assigned_entry_array_.Release();

        memory_heap_entry_list_node_.Release();
        memory_heap_value_.Release();
        CheckCuda(cudaFree(device_->entry_list_head_node_ptrs_memory_pool_));
        CheckCuda(cudaFree(device_->entry_list_size_ptrs_memory_pool_));
    }

    device_ = nullptr;
    bucket_count_ = -1;
    max_value_capacity_ = -1;
    max_linked_list_node_capacity_ = -1;
}

template<typename Key, typename Value, typename Hasher>
void HashTableCuda<Key, Value, Hasher>::UpdateDevice() {
    if (device_ != nullptr) {
        device_->hasher_ = hasher_;
        device_->bucket_count_ = bucket_count_;

        device_->memory_heap_entry_list_node_ =
            *memory_heap_entry_list_node_.device_;
        device_->memory_heap_value_ = *memory_heap_value_.device_;
        device_->entry_array_ = *entry_array_.device_;
        device_->entry_list_array_ = *entry_list_array_.device_;
        device_->lock_array_ = *lock_array_.device_;
        device_->assigned_entry_array_ = *assigned_entry_array_.device_;
    }
}

template<typename Key, typename Value, typename Hasher>
void HashTableCuda<Key, Value, Hasher>::Reset() {
    assert(device_ != nullptr);

    memory_heap_entry_list_node_.Reset();
    memory_heap_value_.Reset();
    ResetEntries();
    ResetLocks();
}

template<typename Key, typename Value, typename Hasher>
void HashTableCuda<Key, Value, Hasher>::ResetEntries() {
    assert(device_ != nullptr);

    HashTableCudaKernelCaller<Key, Value, Hasher>::Reset(*this);
}

template<typename Key, typename Value, typename Hasher>
void HashTableCuda<Key, Value, Hasher>::ResetLocks() {
    assert(device_ != nullptr);

    lock_array_.Memset(0);
}

template<typename Key, typename Value, typename Hasher>
void HashTableCuda<Key, Value, Hasher>::GetAssignedEntries() {
    assert(device_ != nullptr);

    /* Reset counter */
    assigned_entry_array_.Clear();

    HashTableCudaKernelCaller<Key, Value, Hasher>::GetAssignedEntries(*this);
}

template<typename Key, typename Value, typename Hasher>
void HashTableCuda<Key, Value, Hasher>::New(
    std::vector<Key> &keys, std::vector<Value> &values) {
    assert(device_ != nullptr);

    ArrayCuda<Key> keys_cuda(keys.size());
    keys_cuda.Upload(keys);
    ArrayCuda<Value> values_cuda(values.size());
    values_cuda.Upload(values);

    HashTableCudaKernelCaller<Key, Value, Hasher>::Insert(
        *this, keys_cuda, values_cuda, keys.size());
}

template<typename Key, typename Value, typename Hasher>
void HashTableCuda<Key, Value, Hasher>
::Delete(std::vector<Key> &keys) {
    assert(device_ != nullptr);

    ArrayCuda<Key> keys_cuda(keys.size());
    keys_cuda.Upload(keys);

    HashTableCudaKernelCaller<Key, Value, Hasher>::Delete(*this, keys_cuda,
        keys.size());
}

template<typename Key, typename Value, typename Hasher>
std::tuple<std::vector<Key>, std::vector<Value>>
HashTableCuda<Key, Value, Hasher>::Download() {
    assert(device_ != nullptr);

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
    assert(device_ != nullptr);

    std::vector<Entry> ret;
    int assigned_entry_array_size = assigned_entry_array_.size();
    if (assigned_entry_array_size == 0) return ret;
    return assigned_entry_array_.Download();
}

template<typename Key, typename Value, typename Hasher>
std::tuple<std::vector<int>, std::vector<int>>
HashTableCuda<Key, Value, Hasher>::Profile() {
    assert(device_ != nullptr);

    ArrayCuda<int> array_entry_count_cuda(bucket_count_);
    ArrayCuda<int> list_entry_count_cuda(bucket_count_);

    HashTableCudaKernelCaller<Key, Value, Hasher>::Profile(
        *this, array_entry_count_cuda, list_entry_count_cuda);

    std::vector<int> array_entry_count = array_entry_count_cuda.DownloadAll();
    std::vector<int> list_entry_count = list_entry_count_cuda.DownloadAll();

    return std::make_tuple(array_entry_count, list_entry_count);
}
} // cuda
} // open3d