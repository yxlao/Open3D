/*
 * Created by wei on 18-4-2.
 */

#pragma once

#include "HashTableCuda.h"
#include "ArrayCudaDevice.cuh"
#include "LinkedListCudaDevice.cuh"
#include "MemoryHeapCudaDevice.cuh"

#include <cassert>
#include <tuple>
#include <vector>

namespace open3d {

namespace cuda {
/**
 * Server end
 */
template<typename Key, typename Value, typename Hasher>
__device__
int HashTableCudaDevice<Key, Value, Hasher>::GetInternalAddrByKey(
    const Key &key) {
    int bucket_idx = hasher_(key);
    int bucket_base_idx = bucket_idx * BUCKET_SIZE;

    /* 1. Search in the ordered array */
#pragma unroll 1
    for (int i = 0; i < BUCKET_SIZE; ++i) {
        const Entry &entry = entry_array_[bucket_base_idx + i];
        if (entry.Matches(key)) {
            return entry.internal_addr;
        }
    }

    /* 2. Search in the unordered linked list */
    const LinkedListEntryCudaDevice &linked_list =
        entry_list_array_[bucket_idx];
    Entry query_entry;
    query_entry.key = key;
    int entry_node_ptr = linked_list.Find(query_entry);
    if (entry_node_ptr != NULLPTR_CUDA) {
        const Entry &entry = linked_list.get_node(entry_node_ptr).data;
        return entry.internal_addr;
    }

    return NULLPTR_CUDA;
}

template<typename Key, typename Value, typename Hasher>
__device__
    Value
*
HashTableCudaDevice<Key, Value, Hasher>::GetValuePtrByInternalAddr(
    const int addr) {
    return &(memory_heap_value_.value_at(addr));
}

template<typename Key, typename Value, typename Hasher>
__device__
    Value
*
HashTableCudaDevice<Key, Value, Hasher>::GetValuePtrByKey(
    const Key &key) {
    int internal_ptr = GetInternalAddrByKey(key);
    if (internal_ptr == NULLPTR_CUDA) return nullptr;
    return GetValuePtrByInternalAddr(internal_ptr);
}

template<typename Key, typename Value, typename Hasher>
__device__
    Value
*
HashTableCudaDevice<Key, Value, Hasher>::operator[](const Key &key) {
    return GetValuePtrByKey(key);
}

template<typename Key, typename Value, typename Hasher>
__device__
int HashTableCudaDevice<Key, Value, Hasher>::New(
    const Key &key) {
    int bucket_idx = hasher_(key);
    int bucket_base_idx = bucket_idx * BUCKET_SIZE;

    /* 1. Search in the ordered array */
    int entry_array_empty_slot_idx = (-1);
#pragma unroll 1
    for (int i = 0; i < BUCKET_SIZE; ++i) {
        const Entry &entry = entry_array_.at(bucket_base_idx + i);
        if (entry.Matches(key)) {
            return ContainerReturnCode::HashEntryAlreadyExists;
        }
        if (entry_array_empty_slot_idx == (-1)
            && entry.internal_addr == NULLPTR_CUDA) {
            entry_array_empty_slot_idx = bucket_base_idx + i;
        }
    }

    /* 2. Search in the unordered linked list */
    LinkedListEntryCudaDevice &linked_list = entry_list_array_.at(bucket_idx);
    Entry query_entry;
    query_entry.key = key;
    int entry_node_ptr = linked_list.Find(query_entry);
    if (entry_node_ptr != NULLPTR_CUDA) {
        return ContainerReturnCode::HashEntryAlreadyExists;
    }

    /* 3. Both not found. Write to a new entry */
    int lock = atomicExch(&lock_array_.at(bucket_idx), LOCKED);
    if (lock == LOCKED) {
        return ContainerReturnCode::HashEntryIsLocked;
    }

    Entry new_entry;
    new_entry.key = key;
    new_entry.internal_addr = memory_heap_value_.Malloc();

    /* 3.1. Empty slot in ordered part */
    if (entry_array_empty_slot_idx != (-1)) {
        entry_array_.at(entry_array_empty_slot_idx) = new_entry;
    } else { /* 3.2. Insert in the unordered_part */
        linked_list.Insert(new_entry);
    }

    /** Don't unlock, otherwise the result can be inconsistent **/
    return new_entry.internal_addr;
}

template<typename Key, typename Value, typename Hasher>
__device__
int HashTableCudaDevice<Key, Value, Hasher>::Delete(const Key &key) {
    int bucket_idx = hasher_(key);
    int bucket_base_idx = bucket_idx * BUCKET_SIZE;

    /* 1. Search in the ordered array */
#pragma unroll 1
    for (int i = 0; i < BUCKET_SIZE; ++i) {
        Entry &entry = entry_array_.at(bucket_base_idx + i);
        if (entry.Matches(key)) {
            int lock = atomicExch(&lock_array_.at(bucket_idx), LOCKED);
            if (lock == LOCKED) {
                return ContainerReturnCode::HashEntryIsLocked;
            }
            memory_heap_value_.Free(entry.internal_addr);
            entry.Clear();
            return ContainerReturnCode::Success;
        }
    }

    /* 2. Search in the unordered linked list */
    int lock = atomicExch(&lock_array_.at(bucket_idx), LOCKED);
    if (lock == LOCKED) {
        return ContainerReturnCode::HashEntryIsLocked;
    }
    LinkedListEntryCudaDevice &linked_list = entry_list_array_.at(bucket_idx);
    Entry query_entry;
    query_entry.key = key;

    int ret = linked_list.FindAndDelete(query_entry);

    /* An attempt */
    atomicExch(&lock_array_.at(bucket_idx), UNLOCKED);
    return ret;
}
} // cuda
} // open3d