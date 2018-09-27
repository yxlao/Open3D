/**
 * Created by wei on 18-4-2.
 */

#ifndef _HASH_TABLE_CUDA_H_
#define _HASH_TABLE_CUDA_H_

#include "ArrayCuda.h"
#include "LinkedListCuda.h"
#include "MemoryHeapCuda.h"

#include <Cuda/Common/Common.h>
#include <Cuda/Geometry/Vector.h>

namespace three {

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
	__HOSTDEVICE__ size_t operator()(const Vector3i &key) const {
		const int p0 = 73856093;
		const int p1 = 19349669;
		const int p2 = 83492791;

		int r = ((key(0) * p0) ^ (key(1) * p1) ^ (key(2) * p2)) % bucket_count_;
		if (r < 0) r += bucket_count_;
		return (size_t) r;
	}
};

#define BUCKET_SIZE 10

template<typename Key>
class HashEntry {
public:
	Key key;
	int value_ptr;

	__HOSTDEVICE__ bool operator == (const HashEntry<Key>& other) const {
		return key == other.key;
	}
	__HOSTDEVICE__ bool Matches(const Key &other) const {
		return (key == other) && (value_ptr != NULL_PTR);
	}
	__HOSTDEVICE__ bool IsEmpty() {
		return value_ptr == NULL_PTR;
	}
	__HOSTDEVICE__ void Clear() {
		key = Key();
		value_ptr = NULL_PTR;
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
	/* Explicit allocation for LinkedLists living in Array ON GPU */
	int* entry_list_head_node_ptrs_;
	int* entry_list_size_ptrs_;

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

	__DEVICE__ int New(const Key &key);
	__DEVICE__ int Delete(const Key &key);

	__DEVICE__ ArrayCudaServer<Entry> &entry_array() {
		return entry_array_;
	}
	__DEVICE__ ArrayCudaServer<LinkedListEntryCudaServer> &entry_list_array() {
		return entry_list_array_;
	}
	__DEVICE__ ArrayCudaServer<Entry> &assigned_entry_array() {
		return assigned_entry_array_;
	}
	__DEVICE__ MemoryHeapCudaServer<LinkedListNodeEntryCuda>
	    &memory_heap_entry_list_node() {
		return memory_heap_entry_list_node_;
	}
	__DEVICE__ MemoryHeapCudaServer<Value> &memory_heap_value() {
		return memory_heap_value_;
	}
	__DEVICE__ int*& entry_list_head_node_ptrs() {
		return entry_list_head_node_ptrs_;
	}
	__DEVICE__ int*& entry_list_size_ptrs() {
		return entry_list_size_ptrs_;
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
	ArrayCuda<Entry> entry_array_;
	ArrayCuda<LinkedListEntryCudaServer> entry_list_array_;
	ArrayCuda<Entry> assigned_entry_array_;
	MemoryHeapCuda<LinkedListNodeEntryCuda> memory_heap_entry_list_node_;
	MemoryHeapCuda<Value> memory_heap_value_;
	ArrayCuda<int> lock_array_;

	/* Wrap all above */
	HashTableCudaServer<Key, Value, Hasher> server_;

public:
	int bucket_count_;
	int value_max_capacity_;
	int linked_list_node_max_capacity_;

public:
	HashTableCuda() {}
	~HashTableCuda() {};

	void Init(int bucket_count, int value_capacity);
	void Destroy();

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

	HashTableCudaServer<Key, Value, Hasher>& server() {
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
void InitHashTableEntriesKernel(
	HashTableCudaServer<Key, Value, Hasher> server);

template<typename Key, typename Value, typename Hasher>
__GLOBAL__
void DestroyHashTableEntriesKernel(
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
#endif /* _HASH_TABLE_CUDA_H_ */
