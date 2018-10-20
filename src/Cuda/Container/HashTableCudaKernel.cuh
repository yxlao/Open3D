/**
 * Created by wei on 18-4-21.
 */

#include "HashTableCuda.cuh"

#include "LinkedListCuda.h"
#include "MemoryHeapCuda.h"
#include "ArrayCuda.h"

/**
 * Global code called by Host code
 */
namespace open3d {
template<typename Key, typename Value, typename Hasher>
__global__
void CreateHashTableEntriesKernel(
    HashTableCudaServer<Key, Value, Hasher> server) {
    const int bucket_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (bucket_idx >= server.bucket_count_) return;

    int bucket_base_idx = bucket_idx * BUCKET_SIZE;
#pragma unroll 1
    for (int i = 0; i < BUCKET_SIZE; ++i) {
        server.entry_array().get(bucket_base_idx + i).Clear(); /* Clear == Create */
    }

    int *head_node_ptr = &(server.entry_list_head_node_ptrs_memory_pool()[bucket_idx]);
    int *size_ptr = &(server.entry_list_size_ptrs_memory_pool()[bucket_idx]);

    server.entry_list_array().get(bucket_idx).Create(
        server.memory_heap_entry_list_node(),
        head_node_ptr,
        size_ptr);
}

template<typename Key, typename Value, typename Hasher>
__global__
void ReleaseHashTableEntriesKernel(
    HashTableCudaServer<Key, Value, Hasher> server) {
    const int bucket_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (bucket_idx >= server.bucket_count_) return;

    server.entry_list_array().get(bucket_idx).Release();
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
        server.entry_array().get(bucket_base_idx + i).Clear();
    }
    server.entry_list_array().get(bucket_idx).Clear();
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
        Entry &entry = server.entry_array().get(bucket_base_idx + i);
        if (entry.value_ptr != NULL_PTR) {
            server.assigned_entry_array().push_back(entry);
        }
    }

    LinkedListCudaServer<Entry> &linked_list =
        server.entry_list_array().get(bucket_idx);
    int node_ptr = linked_list.head_node_ptr();
    while (node_ptr != NULL_PTR) {
        LinkedListNodeCuda<Entry> &linked_list_node =
            linked_list.get_node(node_ptr);
        server.assigned_entry_array().push_back(linked_list_node.data);
        node_ptr = linked_list_node.next_node_ptr;
    }
}

template<typename Key, typename Value, typename Hasher>
__global__
void InsertHashTableEntriesKernel(
    HashTableCudaServer<Key, Value, Hasher> server,
    Key *keys, Value *values, const int num_pairs) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < num_pairs) {
        int value_internal_ptr = server.New(keys[idx]);

        /* Might fail to allocate when there are thread conflicts */
        if (value_internal_ptr >= 0) {
            Value *value_ptr = server.
                GetValueByInternalValuePtr(value_internal_ptr);
            (*value_ptr) = values[idx];
        }
    }
}

template<typename Key, typename Value, typename Hasher>
__global__
void DeleteHashTableEntriesKernel(
    HashTableCudaServer<Key, Value, Hasher> server,
    Key *keys, const int num_keys) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < num_keys) {
        int ret = server.Delete(keys[idx]);
    }
}

template<typename Key, typename Value, typename Hasher>
__global__
void ProfileHashTableKernel(
    HashTableCudaServer<Key, Value, Hasher> server,
    int *array_entry_count, int *linked_list_entry_count) {
    typedef HashEntry<Key> Entry;

    int bucket_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (bucket_idx >= server.bucket_count_) return;

    int bucket_base_idx = bucket_idx * BUCKET_SIZE;
    int array_entry_cnt = 0;
    for (int i = 0; i < BUCKET_SIZE; ++i) {
        if (!server.entry_array().get(bucket_base_idx + i).IsEmpty()) {
            array_entry_cnt++;
        }
    }

    LinkedListCudaServer<Entry> &linked_list =
        server.entry_list_array().get(bucket_idx);

    int linked_list_entry_cnt = 0;
    int node_ptr = linked_list.head_node_ptr();
    while (node_ptr != NULL_PTR) {
        linked_list_entry_cnt++;
        LinkedListNodeCuda<Entry> &linked_list_node =
            linked_list.get_node(node_ptr);
        node_ptr = linked_list_node.next_node_ptr;
    }

    assert(linked_list.size() == linked_list_entry_cnt);
    array_entry_count[bucket_idx] = array_entry_cnt;
    linked_list_entry_count[bucket_idx] = linked_list_entry_cnt;
}
}