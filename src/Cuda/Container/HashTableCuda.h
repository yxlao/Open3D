/**
 * Created by wei on 18-4-2.
 */

#pragma once

#include "ArrayCuda.h"
#include "LinkedListCuda.h"
#include "MemoryHeapCuda.h"

#include <Cuda/Common/Common.h>
#include <Cuda/Geometry/VectorCuda.h>

#define BUCKET_SIZE 10

namespace open3d {

/**
 * Taken from Niessner, et al, 2013
 * Real-time 3D Reconstruction at Scale using Voxel Hashing
 */
class SpatialHasher {
private:
    int bucket_count_;

public:
    /* Some large default bucket size */
    __HOSTDEVICE__ SpatialHasher() { bucket_count_ = 1000000; }
    __HOSTDEVICE__ SpatialHasher(int bucket_count)
        : bucket_count_(bucket_count) {}
    __HOSTDEVICE__ inline size_t operator()(const Vector3i &key) const {
        const int p0 = 73856093;
        const int p1 = 19349669;
        const int p2 = 83492791;

        int r = ((key(0) * p0) ^ (key(1) * p1) ^ (key(2) * p2)) % bucket_count_;
        if (r < 0) r += bucket_count_;
        return (size_t) r;
    }
};

template<typename Key>
class HashEntry {
public:
    Key key;
    int value_ptr;

    __HOSTDEVICE__ inline bool operator==(const HashEntry<Key> &other) const {
        return key == other.key;
    }
    __HOSTDEVICE__ inline bool Matches(const Key &other) const {
        return (key == other) && (value_ptr != NULLPTR_CUDA);
    }
    __HOSTDEVICE__ inline bool IsEmpty() {
        return value_ptr == NULLPTR_CUDA;
    }
    __HOSTDEVICE__ inline void Clear() {
        key = Key();
        value_ptr = NULLPTR_CUDA;
    }
};

typedef HashEntry<Vector3i> SpatialEntry;

/**
 * My implementation of K\"ahler, et al, 2015
 * Very High Frame Rate Volumetric Integration of Depth Images on Mobile Devices
 *  ordered (array)   unordered (linked list)
 * | | | | | | | --- | | | |
 * | | | | | | | --- | |
 * | | | | | | | --- | | | | | |
 */
template<typename Key, typename Value, typename Hasher>
class HashTableCudaServer {
public:
    typedef HashEntry<Key> Entry;
    typedef LinkedListCudaServer<Entry> LinkedListEntryCudaServer;
    typedef LinkedListNodeCuda<Entry> LinkedListNodeEntryCuda;
    int bucket_count_;

private:
    Hasher hasher_;

    /* bucket_count_ x BUCKET_SIZE */
    ArrayCudaServer<Entry> entry_array_;
    /* bucket_count_ -> LinkedList */
    ArrayCudaServer<LinkedListEntryCudaServer> entry_list_array_;
    /* Collect assigned entries for parallel processing */
    ArrayCudaServer<Entry> assigned_entry_array_;

    ArrayCudaServer<int> lock_array_;

    /* For managing LinkedListNodes and Values */
    MemoryHeapCudaServer<Value> memory_heap_value_;
    MemoryHeapCudaServer<LinkedListNodeEntryCuda> memory_heap_entry_list_node_;

    /** WARNING!!!
      * When our Cuda containers store SERVERS
      * (in this case LinkedListEntryCudaServer),
      * we have to be very careful.
      * - One option is to call Create() for their host classes per element.
      *   but that means, for a 100000 element array, we have to allocate 100000
      *   host classes them on CPU, create them, and push them on GPU
      *   one-by-one. That is too expensive and stupid.
      * - Another option is to allocate them on CUDA using malloc, but that
      *   is very slow (imagine thousands of kernel querying per element
      *   mallocing simultaneously.
      * - So we choose external allocation for LinkedLists, and manage them
      *   on kernels. */
    int *entry_list_head_node_ptrs_memory_pool_;
    int *entry_list_size_ptrs_memory_pool_;

    /** Internal implementations **/
    /**
    * @param key
    * @return ptr (stored in an int addr)
    * that could be accessed in @data_memory_heap_
    * Make it private to avoid confusion
    * between internal ptrs (MemoryHeap) and conventional ptrs (*Object)
    */
public:
    __DEVICE__ int GetInternalValuePtrByKey(const Key &key);
    __DEVICE__ Value *GetValueByInternalValuePtr(const int addr);

    /** External interfaces - return nullable object **/
    __DEVICE__ Value *GetValuePtrByKey(const Key &key);
    __DEVICE__ Value *operator[] (const Key &key);

    __DEVICE__ int New(const Key &key);
    __DEVICE__ int Delete(const Key &key);

    __DEVICE__ inline ArrayCudaServer<Entry> &entry_array() {
        return entry_array_;
    }
    __DEVICE__ inline ArrayCudaServer<LinkedListEntryCudaServer>
        &entry_list_array() {
        return entry_list_array_;
    }
    __DEVICE__ inline ArrayCudaServer<Entry> &assigned_entry_array() {
        return assigned_entry_array_;
    }
    __DEVICE__ inline MemoryHeapCudaServer<LinkedListNodeEntryCuda>
    &memory_heap_entry_list_node() {
        return memory_heap_entry_list_node_;
    }
    __DEVICE__ inline MemoryHeapCudaServer<Value> &memory_heap_value() {
        return memory_heap_value_;
    }
    __DEVICE__ inline int *&entry_list_head_node_ptrs_memory_pool() {
        return entry_list_head_node_ptrs_memory_pool_;
    }
    __DEVICE__ inline int *&entry_list_size_ptrs_memory_pool() {
        return entry_list_size_ptrs_memory_pool_;
    }

    friend class HashTableCuda<Key, Value, Hasher>;
};

template<typename Key, typename Value, typename Hasher>
class HashTableCuda {
public:
    typedef HashEntry<Key> Entry;
    typedef LinkedListCudaServer<Entry> LinkedListEntryCudaServer;
    typedef LinkedListNodeCuda<Entry> LinkedListNodeEntryCuda;

private:
    Hasher hasher_;

    MemoryHeapCuda<LinkedListNodeEntryCuda> memory_heap_entry_list_node_;
    MemoryHeapCuda<Value> memory_heap_value_;

    ArrayCuda<Entry> entry_array_;
    ArrayCuda<LinkedListEntryCudaServer> entry_list_array_;
    ArrayCuda<Entry> assigned_entry_array_;
    ArrayCuda<int> lock_array_;

    /* Wrap all above */
    std::shared_ptr<HashTableCudaServer<Key, Value, Hasher>> server_ = nullptr;

public:
    int bucket_count_;
    int max_value_capacity_;
    int max_linked_list_node_capacity_;

public:
    HashTableCuda();
    HashTableCuda(const HashTableCuda<Key, Value, Hasher>& other);
    HashTableCuda<Key, Value, Hasher>& operator = (
        const HashTableCuda<Key, Value, Hasher> &other);
    ~HashTableCuda();

    void Create(int bucket_count, int value_capacity);
    void Release();
    void UpdateServer();

    void Reset();
    void ResetEntries();
    void ResetLocks();
    void GetAssignedEntries();

    /**
     * The internal data structure is too complicated to be separately dumped.
     * We try to pre-process them before dumping them to CPU.
     * @param pairs
     * @return pair count
     */
    void New(std::vector<Key> &keys, std::vector<Value> &values);
    void Delete(std::vector<Key> &keys);
    std::tuple<std::vector<int>, std::vector<int>> Profile();
    std::tuple<std::vector<Key>, std::vector<Value>> Download();
    std::vector<Entry> DownloadAssignedEntries();

    const Hasher &hasher() const {
        return hasher_;
    }
    int bucket_count() const {
        return bucket_count_;
    }
    int max_value_capacity() const {
        return max_value_capacity_;
    }
    int max_linked_list_node_capacity() const {
        return max_linked_list_node_capacity_;
    }
    const ArrayCuda<Entry>& entry_array() const {
        return entry_array_;
    }
    const ArrayCuda<LinkedListEntryCudaServer> & entry_list_array() const {
        return entry_list_array_;
    }
    const ArrayCuda<Entry>& assigned_entry_array() const {
        return assigned_entry_array_;
    };
    const MemoryHeapCuda<LinkedListNodeEntryCuda>&
        memory_heap_entry_list_node() const {
        return memory_heap_entry_list_node_;
    }
    const MemoryHeapCuda<Value>& memory_heap_value() const {
        return memory_heap_value_;
    }
    const ArrayCuda<int>& lock_array() const {
        return lock_array_;
    }

    std::shared_ptr<HashTableCudaServer<Key, Value, Hasher>> &server() {
        return server_;
    }
    const std::shared_ptr<HashTableCudaServer<Key, Value, Hasher>> &server() const {
        return server_;
    }
};

template<typename Key, typename Value, typename Hasher>
__GLOBAL__
void InsertHashTableEntriesKernel(
    HashTableCudaServer<Key, Value, Hasher> server,
    Key *keys, Value *values, const int num_pairs);

template<typename Key, typename Value, typename Hasher>
__GLOBAL__
void DeleteHashTableEntriesKernel(
    HashTableCudaServer<Key, Value, Hasher> server,
    Key *keys, const int num_keys);

template<typename Key, typename Value, typename Hasher>
__GLOBAL__
void CreateHashTableEntriesKernel(
    HashTableCudaServer<Key, Value, Hasher> server);

template<typename Key, typename Value, typename Hasher>
__GLOBAL__
void ReleaseHashTableEntriesKernel(
    HashTableCudaServer<Key, Value, Hasher> server);

template<typename Key, typename Value, typename Hasher>
__GLOBAL__
void ResetHashTableEntriesKernel(
    HashTableCudaServer<Key, Value, Hasher> server);

template<typename Key, typename Value, typename Hasher>
__GLOBAL__
void GetHashTableAssignedEntriesKernel(
    HashTableCudaServer<Key, Value, Hasher> server);

template<typename Key, typename Value, typename Hasher>
__GLOBAL__
void ProfileHashTableKernel(
    HashTableCudaServer<Key, Value, Hasher> server,
    int *array_entry_count, int *linked_list_entry_count);
}
