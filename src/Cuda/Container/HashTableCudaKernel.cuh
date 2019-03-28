/**
 * Created by wei on 18-4-21.
 */

#pragma once

#include "HashTableCudaDevice.cuh"

/**
 * Global code called by Host code
 */
namespace open3d {

namespace cuda {
template<typename Key, typename Value, typename Hasher>
__global__
void CreateKernel(
    HashTableCudaDevice<Key, Value, Hasher> device) {
    const int bucket_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (bucket_idx >= device.bucket_count_) return;

    int bucket_base_idx = bucket_idx * BUCKET_SIZE;
#pragma unroll 1
    for (int i = 0; i < BUCKET_SIZE; ++i) {
        device.entry_array_.at(bucket_base_idx + i).Clear(); /* Clear == Create */
    }

    int *head_node_ptr = &device.entry_list_head_node_ptrs_memory_pool_[bucket_idx];
    int *size_ptr = &device.entry_list_size_ptrs_memory_pool_[bucket_idx];

    device.entry_list_array_.at(bucket_idx).Create(
        device.memory_heap_entry_list_node_,
        head_node_ptr,
        size_ptr);
}

template<typename Key, typename Value, typename Hasher>
__host__
void HashTableCudaKernelCaller<Key, Value, Hasher>::Create(
    HashTableCuda<Key, Value, Hasher> &hash_table) {
    const int threads = THREAD_1D_UNIT;
    const int blocks = DIV_CEILING(hash_table.bucket_count_, THREAD_1D_UNIT);
    CreateKernel << < blocks, threads >> > (
        *hash_table.device_);
    CheckCuda(cudaDeviceSynchronize());
    CheckCuda(cudaGetLastError());
}

template<typename Key, typename Value, typename Hasher>
__global__
void ReleaseKernel(
    HashTableCudaDevice<Key, Value, Hasher> device) {
    const int bucket_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (bucket_idx >= device.bucket_count_) return;

    device.entry_list_array_.at(bucket_idx).Release();
}

template<typename Key, typename Value, typename Hasher>
__host__
void HashTableCudaKernelCaller<Key, Value, Hasher>::Release(
    HashTableCuda<Key, Value, Hasher> &hash_table) {

    const int blocks = DIV_CEILING(hash_table.bucket_count_, THREAD_1D_UNIT);
    const int threads = THREAD_1D_UNIT;
    ReleaseKernel<< < blocks, threads >> >(*hash_table.device_);
    CheckCuda(cudaDeviceSynchronize());
    CheckCuda(cudaGetLastError());
}

template<typename Key, typename Value, typename Hasher>
__global__
void ResetKernel(HashTableCudaDevice<Key, Value, Hasher> device) {
    const int bucket_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (bucket_idx >= device.bucket_count_) return;

    int bucket_base_idx = bucket_idx * BUCKET_SIZE;
#pragma unroll 1
    for (int i = 0; i < BUCKET_SIZE; ++i) {
        device.entry_array_.at(bucket_base_idx + i).Clear();
    }
    device.entry_list_array_.at(bucket_idx).Clear();
}

template<typename Key, typename Value, typename Hasher>
__host__
void HashTableCudaKernelCaller<Key, Value, Hasher>::Reset(
    HashTableCuda<Key, Value, Hasher> &hash_table) {
    const int blocks = DIV_CEILING(hash_table.bucket_count_, THREAD_1D_UNIT);
    const int threads = THREAD_1D_UNIT;
    ResetKernel << < blocks, threads >> > (*hash_table.device_);
    CheckCuda(cudaDeviceSynchronize());
    CheckCuda(cudaGetLastError());
}

template<typename Key, typename Value, typename Hasher>
__global__
void GetAssignedEntriesKernel(
    HashTableCudaDevice<Key, Value, Hasher> device) {
    typedef HashEntry<Key> Entry;

    const int bucket_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (bucket_idx >= device.bucket_count_) return;

    int bucket_base_idx = bucket_idx * BUCKET_SIZE;
#pragma unroll 1
    for (int i = 0; i < BUCKET_SIZE; ++i) {
        Entry &entry = device.entry_array_.at(bucket_base_idx + i);
        if (entry.internal_addr != NULLPTR_CUDA) {
            device.assigned_entry_array_.push_back(entry);
        }
    }

    LinkedListCudaDevice<Entry> &linked_list =
        device.entry_list_array_.at(bucket_idx);
    int node_ptr = linked_list.head_node_ptr();
    while (node_ptr != NULLPTR_CUDA) {
        LinkedListNodeCuda<Entry> &linked_list_node =
            linked_list.get_node(node_ptr);
        device.assigned_entry_array_.push_back(linked_list_node.data);
        node_ptr = linked_list_node.next_node_ptr;
    }
}

template<typename Key, typename Value, typename Hasher>
__HOST__
void HashTableCudaKernelCaller<Key, Value, Hasher>::GetAssignedEntries(
    HashTableCuda<Key, Value, Hasher> &hash_table) {

    const int blocks = DIV_CEILING(hash_table.bucket_count_, THREAD_1D_UNIT);
    const int threads = THREAD_1D_UNIT;
    GetAssignedEntriesKernel << < blocks, threads >> > (
        *hash_table.device_);
    CheckCuda(cudaDeviceSynchronize());
    CheckCuda(cudaGetLastError());
}

template<typename Key, typename Value, typename Hasher>
__global__
void InsertKernel(
    HashTableCudaDevice<Key, Value, Hasher> device,
    ArrayCudaDevice<Key> keys, ArrayCudaDevice<Value> values,
    int num_pairs) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < num_pairs) {
        int value_internal_ptr = device.New(keys[idx]);

        /* Might fail to allocate when there are thread conflicts */
        if (value_internal_ptr >= 0) {
            Value *value_ptr = device.
                GetValuePtrByInternalAddr(value_internal_ptr);
            (*value_ptr) = values[idx];
        }
    }
}

template<typename Key, typename Value, typename Hasher>
__host__
void HashTableCudaKernelCaller<Key, Value, Hasher>::Insert(
    HashTableCuda<Key, Value, Hasher> &hash_table,
    ArrayCuda<Key> &keys, ArrayCuda<Value> &values,
    int num_pairs) {
    const int blocks = DIV_CEILING(hash_table.bucket_count_, THREAD_1D_UNIT);
    const int threads = THREAD_1D_UNIT;
    InsertKernel<< < blocks, threads >> > (
        *hash_table.device_,
        *keys.device_, *values.device_, num_pairs);
    CheckCuda(cudaDeviceSynchronize());
    CheckCuda(cudaGetLastError());
}

template<typename Key, typename Value, typename Hasher>
__global__
void DeleteKernel(
    HashTableCudaDevice<Key, Value, Hasher> device,
    ArrayCudaDevice<Key> keys, int num_keys) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < num_keys) {
        int ret = device.Delete(keys[idx]);
    }
}

template<typename Key, typename Value, typename Hasher>
__host__
void HashTableCudaKernelCaller<Key, Value, Hasher>::Delete(
    HashTableCuda<Key, Value, Hasher> &hash_table,
    ArrayCuda<Key> &keys, int num_keys) {
    const int blocks = DIV_CEILING(hash_table.bucket_count_, THREAD_1D_UNIT);
    const int threads = THREAD_1D_UNIT;

    DeleteKernel << < blocks, threads >> > (*hash_table.device_,
        *keys.device_, num_keys);
    CheckCuda(cudaDeviceSynchronize());
    CheckCuda(cudaGetLastError());
}

template<typename Key, typename Value, typename Hasher>
__global__
void ProfileKernel(
    HashTableCudaDevice<Key, Value, Hasher> device,
    ArrayCudaDevice<int> array_entry_count,
    ArrayCudaDevice<int> linked_list_entry_count) {
    typedef HashEntry<Key> Entry;

    int bucket_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (bucket_idx >= device.bucket_count_) return;

    int bucket_base_idx = bucket_idx * BUCKET_SIZE;
    int array_entry_cnt = 0;
    for (int i = 0; i < BUCKET_SIZE; ++i) {
        if (!device.entry_array_.at(bucket_base_idx + i).IsEmpty()) {
            array_entry_cnt++;
        }
    }

    LinkedListCudaDevice<Entry> &linked_list =
        device.entry_list_array_.at(bucket_idx);

    int linked_list_entry_cnt = 0;
    int node_ptr = linked_list.head_node_ptr();
    while (node_ptr != NULLPTR_CUDA) {
        linked_list_entry_cnt++;
        LinkedListNodeCuda<Entry> &linked_list_node =
            linked_list.get_node(node_ptr);
        node_ptr = linked_list_node.next_node_ptr;
    }

    assert(linked_list.size() == linked_list_entry_cnt);
    array_entry_count[bucket_idx] = array_entry_cnt;
    linked_list_entry_count[bucket_idx] = linked_list_entry_cnt;
}

template<typename Key, typename Value, typename Hasher>
__host__
void HashTableCudaKernelCaller<Key, Value, Hasher>::Profile(
    HashTableCuda<Key, Value, Hasher> &hash_table,
    ArrayCuda<int> &array_entry_count,
    ArrayCuda<int> &linked_list_entry_count) {
    const int threads = THREAD_1D_UNIT;
    const int blocks = DIV_CEILING(hash_table.bucket_count_, THREAD_1D_UNIT);
    ProfileKernel << < blocks, threads >> > (*hash_table.device_,
        *array_entry_count.device_, *linked_list_entry_count.device_);
    CheckCuda(cudaDeviceSynchronize());
    CheckCuda(cudaGetLastError());
}
} // cuda
} // open3d