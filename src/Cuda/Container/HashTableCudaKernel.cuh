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
void CreateHashTableEntriesKernel(
    HashTableCudaServer<Key, Value, Hasher> server) {
    const int bucket_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (bucket_idx >= server.bucket_count_) return;

    int bucket_base_idx = bucket_idx * BUCKET_SIZE;
#pragma unroll 1
    for (int i = 0; i < BUCKET_SIZE; ++i) {
        server.entry_array().at(
            bucket_base_idx + i).Clear(); /* Clear == Create */
    }

    int *head_node_ptr =
        &(server.entry_list_head_node_ptrs_memory_pool()[bucket_idx]);
    int *size_ptr = &(server.entry_list_size_ptrs_memory_pool()[bucket_idx]);

    server.entry_list_array().at(bucket_idx).Create(
        server.memory_heap_entry_list_node(),
        head_node_ptr,
        size_ptr);
}

template<typename Key, typename Value, typename Hasher>
__host__
void HashTableCudaKernelCaller<Key, Value, Hasher>::
CreateHashTableEntriesKernelCaller(
    HashTableCudaServer<Key, Value, Hasher> &server,
    int bucket_count) {
    const int threads = THREAD_1D_UNIT;
    const int blocks = DIV_CEILING(bucket_count, THREAD_1D_UNIT);
    CreateHashTableEntriesKernel << < blocks, threads >> > (server);
    CheckCuda(cudaDeviceSynchronize());
    CheckCuda(cudaGetLastError());
}

template<typename Key, typename Value, typename Hasher>
__global__
void ReleaseHashTableEntriesKernel(
    HashTableCudaServer<Key, Value, Hasher> server) {
    const int bucket_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (bucket_idx >= server.bucket_count_) return;

    server.entry_list_array().at(bucket_idx).Release();
}

template<typename Key, typename Value, typename Hasher>
__host__
void HashTableCudaKernelCaller<Key, Value, Hasher>::
ReleaseHashTableEntriesKernelCaller(
    HashTableCudaServer<Key, Value, Hasher> &server,
    int bucket_count) {

    const int blocks = DIV_CEILING(bucket_count, THREAD_1D_UNIT);
    const int threads = THREAD_1D_UNIT;

    ReleaseHashTableEntriesKernel << < blocks, threads >> > (server);
    CheckCuda(cudaDeviceSynchronize());
    CheckCuda(cudaGetLastError());
}

template<typename Key, typename Value, typename Hasher>
__global__
void ResetHashTableEntriesKernel(
    HashTableCudaServer<Key, Value, Hasher> server) {
    const int bucket_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (bucket_idx >= server.bucket_count_) return;

    int bucket_base_idx = bucket_idx * BUCKET_SIZE;
#pragma unroll 1
    for (int i = 0; i < BUCKET_SIZE; ++i) {
        server.entry_array().at(bucket_base_idx + i).Clear();
    }
    server.entry_list_array().at(bucket_idx).Clear();
}

template<typename Key, typename Value, typename Hasher>
__host__
void HashTableCudaKernelCaller<Key, Value, Hasher>::
ResetHashTableEntriesKernelCaller(
    HashTableCudaServer<Key, Value, Hasher> &server,
    int bucket_count) {
    const int blocks = DIV_CEILING(bucket_count, THREAD_1D_UNIT);
    const int threads = THREAD_1D_UNIT;
    ResetHashTableEntriesKernel << < blocks, threads >> > (server);
    CheckCuda(cudaDeviceSynchronize());
    CheckCuda(cudaGetLastError());

}

template<typename Key, typename Value, typename Hasher>
__global__
void GetHashTableAssignedEntriesKernel(
    HashTableCudaServer<Key, Value, Hasher> server) {
    typedef HashEntry<Key> Entry;

    const int bucket_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (bucket_idx >= server.bucket_count_) return;

    int bucket_base_idx = bucket_idx * BUCKET_SIZE;
#pragma unroll 1
    for (int i = 0; i < BUCKET_SIZE; ++i) {
        Entry &entry = server.entry_array().at(bucket_base_idx + i);
        if (entry.internal_addr != NULLPTR_CUDA) {
            server.assigned_entry_array().push_back(entry);
        }
    }

    LinkedListCudaServer<Entry> &linked_list =
        server.entry_list_array().at(bucket_idx);
    int node_ptr = linked_list.head_node_ptr();
    while (node_ptr != NULLPTR_CUDA) {
        LinkedListNodeCuda<Entry> &linked_list_node =
            linked_list.get_node(node_ptr);
        server.assigned_entry_array().push_back(linked_list_node.data);
        node_ptr = linked_list_node.next_node_ptr;
    }
}

template<typename Key, typename Value, typename Hasher>
__HOST__
void HashTableCudaKernelCaller<Key, Value, Hasher>::
GetHashTableAssignedEntriesKernelCaller(
    HashTableCudaServer<Key, Value, Hasher> &server,
    int bucket_count) {

    const int blocks = DIV_CEILING(bucket_count, THREAD_1D_UNIT);
    const int threads = THREAD_1D_UNIT;
    GetHashTableAssignedEntriesKernel << < blocks, threads >> > (server);
    CheckCuda(cudaDeviceSynchronize());
    CheckCuda(cudaGetLastError());
}

template<typename Key, typename Value, typename Hasher>
__global__
void InsertHashTableEntriesKernel(
    HashTableCudaServer<Key, Value, Hasher> server,
    ArrayCudaServer<Key> keys, ArrayCudaServer<Value> values, int num_pairs) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < num_pairs) {
        int value_internal_ptr = server.New(keys[idx]);

        /* Might fail to allocate when there are thread conflicts */
        if (value_internal_ptr >= 0) {
            Value *value_ptr = server.
                GetValuePtrByInternalAddr(value_internal_ptr);
            (*value_ptr) = values[idx];
        }
    }
}

template<typename Key, typename Value, typename Hasher>
__host__
void HashTableCudaKernelCaller<Key, Value, Hasher>::
InsertHashTableEntriesKernelCaller(
    HashTableCudaServer<Key, Value, Hasher> &server,
    ArrayCudaServer<Key> &keys,
    ArrayCudaServer<Value> &values,
    int num_pairs,
    int bucket_count) {
    const int blocks = DIV_CEILING(bucket_count, THREAD_1D_UNIT);
    const int threads = THREAD_1D_UNIT;
    InsertHashTableEntriesKernel << < blocks, threads >> > (server,
        keys, values, num_pairs);
    CheckCuda(cudaDeviceSynchronize());
    CheckCuda(cudaGetLastError());
}

template<typename Key, typename Value, typename Hasher>
__global__
void DeleteHashTableEntriesKernel(
    HashTableCudaServer<Key, Value, Hasher> server,
    ArrayCudaServer<Key> keys, int num_keys) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < num_keys) {
        int ret = server.Delete(keys[idx]);
    }
}

template<typename Key, typename Value, typename Hasher>
__host__
void HashTableCudaKernelCaller<Key, Value, Hasher>::
DeleteHashTableEntriesKernelCaller(
    HashTableCudaServer<Key, Value, Hasher> &server,
    ArrayCudaServer<Key> &keys,
    int num_keys,
    int bucket_count) {
    const int blocks = DIV_CEILING(bucket_count, THREAD_1D_UNIT);
    const int threads = THREAD_1D_UNIT;

    DeleteHashTableEntriesKernel << < blocks, threads >> > (server,
        keys, num_keys);
    CheckCuda(cudaDeviceSynchronize());
    CheckCuda(cudaGetLastError());
}

template<typename Key, typename Value, typename Hasher>
__global__
void ProfileHashTableKernel(
    HashTableCudaServer<Key, Value, Hasher> server,
    ArrayCudaServer<int> array_entry_count,
    ArrayCudaServer<int> linked_list_entry_count) {
    typedef HashEntry<Key> Entry;

    int bucket_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (bucket_idx >= server.bucket_count_) return;

    int bucket_base_idx = bucket_idx * BUCKET_SIZE;
    int array_entry_cnt = 0;
    for (int i = 0; i < BUCKET_SIZE; ++i) {
        if (!server.entry_array().at(bucket_base_idx + i).IsEmpty()) {
            array_entry_cnt++;
        }
    }

    LinkedListCudaServer<Entry> &linked_list =
        server.entry_list_array().at(bucket_idx);

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
void HashTableCudaKernelCaller<Key, Value, Hasher>::
ProfileHashTableKernelCaller(
    HashTableCudaServer<Key, Value, Hasher> &server,
    ArrayCudaServer<int> &array_entry_count,
    ArrayCudaServer<int> &linked_list_entry_count,
    int bucket_count) {
    const int threads = THREAD_1D_UNIT;
    const int blocks = DIV_CEILING(bucket_count, THREAD_1D_UNIT);
    ProfileHashTableKernel << < blocks, threads >> > (server,
        array_entry_count, linked_list_entry_count);
    CheckCuda(cudaDeviceSynchronize());
    CheckCuda(cudaGetLastError());
}
} // cuda
} // open3d