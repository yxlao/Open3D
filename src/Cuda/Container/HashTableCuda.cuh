/*
 * Created by wei on 18-4-2.
 */

#pragma once

#include "HashTableCuda.h"
#include "ArrayCuda.cuh"
#include "LinkedListCuda.cuh"
#include "MemoryHeapCuda.cuh"

#include <Cuda/Common/Common.h>
#include <cuda.h>

#include <vector>
#include <cassert>
#include <tuple>

namespace open3d {
/**
 * Server end
 */
template<typename Key, typename Value, typename Hasher>
__device__
int HashTableCudaServer<Key, Value, Hasher>::GetInternalValuePtrByKey(
    const Key &key) {
    int bucket_idx = hasher_(key);
    int bucket_base_idx = bucket_idx * BUCKET_SIZE;

    /* 1. Search in the ordered array */
#pragma unroll 1
    for (int i = 0; i < BUCKET_SIZE; ++i) {
        const Entry &entry = entry_array_[bucket_base_idx + i];
        if (entry.Matches(key)) {
            return entry.value_ptr;
        }
    }

    /* 2. Search in the unordered linked list */
    const LinkedListEntryCudaServer
        &linked_list = entry_list_array_[bucket_idx];
    Entry query_entry;
    query_entry.key = key;
    int entry_node_ptr = linked_list.Find(query_entry);
    if (entry_node_ptr != LINKED_LIST_NODE_NOT_FOUND) {
        const Entry &entry = linked_list.get_node(entry_node_ptr).data;
        return entry.value_ptr;
    }

    return NULLPTR_CUDA;
}

template<typename Key, typename Value, typename Hasher>
__device__
Value *HashTableCudaServer<Key, Value, Hasher>::GetValueByInternalValuePtr(
    const int addr) {
    return &(memory_heap_value_.get_value(addr));
}

template<typename Key, typename Value, typename Hasher>
__device__
Value *HashTableCudaServer<Key, Value, Hasher>
::GetValuePtrByKey(const Key &key) {
    int internal_ptr = GetInternalValuePtrByKey(key);
    if (internal_ptr == NULLPTR_CUDA) return nullptr;
    return GetValueByInternalValuePtr(internal_ptr);
}

template<typename Key, typename Value, typename Hasher>
__device__
Value *HashTableCudaServer<Key, Value, Hasher>::operator[] (const Key &key) {
    return GetValuePtrByKey(key);
}

template<typename Key, typename Value, typename Hasher>
__device__
int HashTableCudaServer<Key, Value, Hasher>::New(const Key &key) {
    int bucket_idx = hasher_(key);
    int bucket_base_idx = bucket_idx * BUCKET_SIZE;

    /* 1. Search in the ordered array */
    int entry_array_empty_slot_idx = (-1);
#pragma unroll 1
    for (int i = 0; i < BUCKET_SIZE; ++i) {
        const Entry &entry = entry_array_.get(bucket_base_idx + i);
        if (entry.Matches(key)) {
            return HASH_ENTRY_EXISTING;
        }
        if (entry_array_empty_slot_idx == (-1)
            && entry.value_ptr == HASH_ENTRY_EMPTY) {
            entry_array_empty_slot_idx = bucket_base_idx + i;
        }
    }

    /* 2. Search in the unordered linked list */
    LinkedListEntryCudaServer &linked_list = entry_list_array_.get(bucket_idx);
    Entry query_entry;
    query_entry.key = key;
    int entry_node_ptr = linked_list.Find(query_entry);
    if (entry_node_ptr != LINKED_LIST_NODE_NOT_FOUND) {
        return HASH_ENTRY_EXISTING;
    }

    /* 3. Both not found. Write to a new entry */
    int lock = atomicExch(&lock_array_.get(bucket_idx), LOCKED);
    if (lock == LOCKED) {
        return HASH_ENTRY_LOCKED;
    }

    Entry new_entry;
    new_entry.key = key;
    new_entry.value_ptr = memory_heap_value_.Malloc();

    /* 3.1. Empty slot in ordered part */
    if (entry_array_empty_slot_idx != (-1)) {
        entry_array_.get(entry_array_empty_slot_idx) = new_entry;
    } else { /* 3.2. Insert in the unordered_part */
        linked_list.Insert(new_entry);
    }

    /* An attempt */
    atomicExch(&lock_array_.get(bucket_idx), UNLOCKED);
    return new_entry.value_ptr;
}

template<typename Key, typename Value, typename Hasher>
__device__
int HashTableCudaServer<Key, Value, Hasher>::Delete(const Key &key) {
    int bucket_idx = hasher_(key);
    int bucket_base_idx = bucket_idx * BUCKET_SIZE;

    /* 1. Search in the ordered array */
#pragma unroll 1
    for (int i = 0; i < BUCKET_SIZE; ++i) {
        Entry &entry = entry_array_.get(bucket_base_idx + i);
        if (entry.Matches(key)) {
            int lock = atomicExch(&lock_array_.get(bucket_idx), LOCKED);
            if (lock == LOCKED) {
                return HASH_ENTRY_LOCKED;
            }
            memory_heap_value_.Free(entry.value_ptr);
            entry.Clear();
            return SUCCESS;
        }
    }

    /* 2. Search in the unordered linked list */
    int lock = atomicExch(&lock_array_.get(bucket_idx), LOCKED);
    if (lock == LOCKED) {
        return HASH_ENTRY_LOCKED;
    }
    LinkedListEntryCudaServer &linked_list = entry_list_array_.get(bucket_idx);
    Entry query_entry;
    query_entry.key = key;

    int ret = linked_list.FindAndDelete(query_entry);

    /* An attempt */
    atomicExch(&lock_array_.get(bucket_idx), UNLOCKED);
    return ret;
}

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

    bucket_count_ = other.bucket_count();
    max_value_capacity_ = other.max_value_capacity();
    max_linked_list_node_capacity_ = other.max_linked_list_node_capacity();
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
        bucket_count_ = other.bucket_count();
        max_value_capacity_ = other.max_value_capacity();
        max_linked_list_node_capacity_ = other.max_linked_list_node_capacity();
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
        PrintError("Already created, stop re-creating!\n");
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

    const int threads = THREAD_1D_UNIT;
    const int blocks = DIV_CEILING(bucket_count, THREAD_1D_UNIT);
    CreateHashTableEntriesKernel << < blocks, threads >> > (*server_);
    CheckCuda(cudaDeviceSynchronize());
    CheckCuda(cudaGetLastError());
}

template<typename Key, typename Value, typename Hasher>
void HashTableCuda<Key, Value, Hasher>::Release() {
    /** Since we Release, we don't care about the content of the linked list
     * array. They are stored in the memory_heap and will be Releaseed anyway.
     */
    if (server_ != nullptr && server_.use_count() == 1) {
        const int threads = THREAD_1D_UNIT;
        const int blocks = DIV_CEILING(bucket_count_, THREAD_1D_UNIT);
        ReleaseHashTableEntriesKernel << < blocks, threads >> > (*server_);
        CheckCuda(cudaDeviceSynchronize());

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

template<typename Key, typename Value, typename Hasher>
void HashTableCuda<Key, Value, Hasher>::Reset() {
    memory_heap_entry_list_node_.Reset();
    memory_heap_value_.Reset();
    ResetEntries();
    ResetLocks();
}

template<typename Key, typename Value, typename Hasher>
void HashTableCuda<Key, Value, Hasher>::ResetEntries() {
    const int threads = THREAD_1D_UNIT;
    const int blocks = DIV_CEILING(bucket_count_, THREAD_1D_UNIT);
    ResetHashTableEntriesKernel << < blocks, threads >> > (*server_);
    CheckCuda(cudaDeviceSynchronize());
    CheckCuda(cudaGetLastError());
}

template<typename Key, typename Value, typename Hasher>
void HashTableCuda<Key, Value, Hasher>::ResetLocks() {
    lock_array_.Memset(0);
}

template<typename Key, typename Value, typename Hasher>
void HashTableCuda<Key, Value, Hasher>::GetAssignedEntries() {
    /* Reset counter */
    assigned_entry_array_.Clear();

    const int threads = THREAD_1D_UNIT;
    const int blocks = DIV_CEILING(bucket_count_, THREAD_1D_UNIT);
    GetHashTableAssignedEntriesKernel << < blocks, threads >> > (*server_);
    CheckCuda(cudaDeviceSynchronize());
}

template<typename Key, typename Value, typename Hasher>
void HashTableCuda<Key, Value, Hasher>::New(
    std::vector<Key> &keys, std::vector<Value> &values) {
    Key *keys_packets;
    Value *values_packets;
    CheckCuda(cudaMalloc(&keys_packets, sizeof(Key) * keys.size()));
    CheckCuda(cudaMemcpy(keys_packets, keys.data(),
                         sizeof(Key) * keys.size(),
                         cudaMemcpyHostToDevice));
    CheckCuda(cudaMalloc(&values_packets, sizeof(Value) * values.size()));
    CheckCuda(cudaMemcpy(values_packets, values.data(),
                         sizeof(Value) * values.size(),
                         cudaMemcpyHostToDevice));

    const int threads = THREAD_1D_UNIT;
    const int blocks = DIV_CEILING(bucket_count_, THREAD_1D_UNIT);
    InsertHashTableEntriesKernel << < blocks, threads >> > (
        *server_, keys_packets, values_packets, keys.size());
    CheckCuda(cudaDeviceSynchronize());
    CheckCuda(cudaGetLastError());

    CheckCuda(cudaFree(keys_packets));
    CheckCuda(cudaFree(values_packets));
}

template<typename Key, typename Value, typename Hasher>
void HashTableCuda<Key, Value, Hasher>
::Delete(std::vector<Key> &keys) {

    Key *keys_packets;
    CheckCuda(cudaMalloc(&keys_packets, sizeof(Key) * keys.size()));
    CheckCuda(cudaMemcpy(keys_packets, keys.data(),
                         sizeof(Key) * keys.size(),
                         cudaMemcpyHostToDevice));

    const int threads = THREAD_1D_UNIT;
    const int blocks = DIV_CEILING(bucket_count_, THREAD_1D_UNIT);
    DeleteHashTableEntriesKernel << < blocks, threads >> > (
        *server_, keys_packets, keys.size());
    CheckCuda(cudaDeviceSynchronize());
    CheckCuda(cudaGetLastError());

    CheckCuda(cudaFree(keys_packets));
}

template<typename Key, typename Value, typename Hasher>
std::tuple<std::vector<Key>, std::vector<Value>>
HashTableCuda<Key, Value, Hasher>::Download() {
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
        values[i] = memory_heap_values[entry.value_ptr];
    }

    return std::make_tuple(keys, values);
}

template<typename Key, typename Value, typename Hasher>
std::vector<HashEntry<Key>> HashTableCuda<Key, Value, Hasher>
::DownloadAssignedEntries() {
    std::vector<Entry> ret;
    int assigned_entry_array_size = assigned_entry_array_.size();
    if (assigned_entry_array_size == 0) return ret;
    return assigned_entry_array_.Download();
}

template<typename Key, typename Value, typename Hasher>
std::tuple<std::vector<int>, std::vector<int>>
HashTableCuda<Key, Value, Hasher>::Profile() {

    std::vector<int> array_entry_count;
    std::vector<int> list_entry_count;
    array_entry_count.resize(bucket_count_);
    list_entry_count.resize(bucket_count_);

    int *array_entry_count_packets;
    int *list_entry_count_packets;
    CheckCuda(cudaMalloc(&array_entry_count_packets,
                         sizeof(int) * bucket_count_));
    CheckCuda(cudaMalloc(&list_entry_count_packets,
                         sizeof(int) * bucket_count_));

    const int threads = THREAD_1D_UNIT;
    const int blocks = DIV_CEILING(bucket_count_, THREAD_1D_UNIT);
    ProfileHashTableKernel << < blocks, threads >> > (
        *server_, array_entry_count_packets, list_entry_count_packets);
    CheckCuda(cudaDeviceSynchronize());
    CheckCuda(cudaGetLastError());

    CheckCuda(cudaMemcpy(array_entry_count.data(), array_entry_count_packets,
                         sizeof(int) * bucket_count_, cudaMemcpyDeviceToHost));
    CheckCuda(cudaMemcpy(list_entry_count.data(), list_entry_count_packets,
                         sizeof(int) * bucket_count_, cudaMemcpyDeviceToHost));
    CheckCuda(cudaFree(array_entry_count_packets));
    CheckCuda(cudaFree(list_entry_count_packets));

    return std::make_tuple(array_entry_count, list_entry_count);
}
}