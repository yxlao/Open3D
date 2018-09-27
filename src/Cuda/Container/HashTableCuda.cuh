/*
 * Created by wei on 18-4-2.
 */

#ifndef _HASH_TABLE_CUDA_CUH_
#define _HASH_TABLE_CUDA_CUH_

#include "HashTableCuda.h"
#include "ArrayCuda.cuh"
#include "LinkedListCuda.cuh"
#include "MemoryHeapCuda.cuh"

#include <Cuda/Common/Common.h>

#include <vector>
#include <cassert>
#include <tuple>

namespace three {
/**
 * Server end
 */
template<typename Key, typename Value, typename Hasher>
__device__
int HashTableCudaServer<Key, Value, Hasher>
    ::GetInternalValuePtrByKey(const Key &key) {
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
	const LinkedListEntryCudaServer &linked_list = entry_list_array_[bucket_idx];
	Entry query_entry;
	query_entry.key = key;
	int entry_node_ptr = linked_list.Find(query_entry);
	if (entry_node_ptr != NODE_NOT_FOUND) {
		const Entry &entry = linked_list.get_node(entry_node_ptr).data;
		return entry.value_ptr;
	}

	return NULL_PTR;
}

template<typename Key, typename Value, typename Hasher>
__device__
Value* HashTableCudaServer<Key, Value, Hasher>
    ::GetValueByInternalValuePtr(const int addr) {
	return &(memory_heap_value_.get_value(addr));
}

template<typename Key, typename Value, typename Hasher>
__device__
Value* HashTableCudaServer<Key, Value, Hasher>
    ::GetValuePtrByKey(const Key &key) {
	int internal_ptr = GetInternalValuePtrByKey(key);
	if (internal_ptr == NULL_PTR) return nullptr;
	return GetValueByInternalValuePtr(internal_ptr);
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
			return ENTRY_EXISTED;
		}
		if (entry_array_empty_slot_idx == (-1)
			&& entry.value_ptr == EMPTY_ENTRY) {
			entry_array_empty_slot_idx = bucket_base_idx + i;
		}
	}

	/* 2. Search in the unordered linked list */
	LinkedListEntryCudaServer &linked_list = entry_list_array_.get(bucket_idx);
	Entry query_entry;
	query_entry.key = key;
	int entry_node_ptr = linked_list.Find(query_entry);
	if (entry_node_ptr != NODE_NOT_FOUND) {
		return ENTRY_EXISTED;
	}

	/* 3. Both not found. Write to a new entry */
	int lock = atomicExch(&lock_array_.get(bucket_idx), LOCKED);
	if (lock == LOCKED) {
		return LOCKED;
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
				return LOCKED;
			}
			memory_heap_value_.Free(entry.value_ptr);
			entry.Clear();
			return SUCCESS;
		}
	}

	/* 2. Search in the unordered linked list */
	int lock = atomicExch(&lock_array_.get(bucket_idx), LOCKED);
	if (lock == LOCKED) {
		return LOCKED;
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
void HashTableCuda<Key, Value, Hasher>::Init(
	int bucket_count, int value_capacity) {
	bucket_count_ = bucket_count; /* (* BUCKET_SIZE), 2D array */
	hasher_ = Hasher(bucket_count);

	value_max_capacity_ = value_capacity;
	linked_list_node_max_capacity_ = bucket_count * BUCKET_SIZE;

	memory_heap_entry_list_node_.Init(linked_list_node_max_capacity_);
	memory_heap_value_.Init(value_max_capacity_);

	entry_array_.Init(bucket_count * BUCKET_SIZE);
	entry_list_array_.Init(bucket_count);
	lock_array_.Init(bucket_count);
	assigned_entry_array_.Init(
		bucket_count * BUCKET_SIZE + linked_list_node_max_capacity_);

	server_.hasher_ = hasher_;
	server_.bucket_count_ = bucket_count;
	server_.memory_heap_entry_list_node_ =
		memory_heap_entry_list_node_.server();
	server_.memory_heap_value_ = memory_heap_value_.server();
	server_.entry_array_ = entry_array_.server();
	server_.entry_list_array_ = entry_list_array_.server();
	server_.lock_array_ = lock_array_.server();
	server_.assigned_entry_array_ = assigned_entry_array_.server();

	CudaMalloc((void**)&server_.entry_list_head_node_ptrs_,
		sizeof(int) * bucket_count);
	CudaMalloc((void**)&server_.entry_list_size_ptrs_,
		sizeof(int) * bucket_count);

	const int threads = THREAD_1D_UNIT;
	const int blocks = UPPER_ALIGN(bucket_count, THREAD_1D_UNIT);
	InitHashTableEntriesKernel<<<blocks, threads>>>(server_);
	CudaSynchronize();
}

template<typename Key, typename Value, typename Hasher>
void HashTableCuda<Key, Value, Hasher>::Destroy() {
	/** Since we Destroy, we don't care about the content of the linked list
	 * array. They are stored in the memory_heap and will be destroyed anyway.
	 */
	const int threads = THREAD_1D_UNIT;
	const int blocks = UPPER_ALIGN(bucket_count_, THREAD_1D_UNIT);
	DestroyHashTableEntriesKernel<<<blocks, threads>>>(server_);
	CudaSynchronize();

	entry_array_.Destroy();
	entry_list_array_.Destroy();
	lock_array_.Destroy();
	assigned_entry_array_.Destroy();

	memory_heap_entry_list_node_.Destroy();
	memory_heap_value_.Destroy();
	CudaFree(server_.entry_list_head_node_ptrs_);
	CudaFree(server_.entry_list_size_ptrs_);
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
	const int blocks = UPPER_ALIGN(bucket_count_, THREAD_1D_UNIT);
	ResetHashTableEntriesKernel<<<blocks, threads>>>(server_);
	CudaSynchronize();
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
	const int blocks = UPPER_ALIGN(bucket_count_, THREAD_1D_UNIT);
	GetHashTableAssignedEntriesKernel<<< blocks, threads>>>(server_);
	CudaSynchronize();
}

template<typename Key, typename Value, typename Hasher>
void HashTableCuda<Key, Value, Hasher>
    ::New(std::vector<Key> &keys, std::vector<Value> &values) {
    Key *keys_packets;
    Value *values_packets;
    CudaMalloc((void**)&keys_packets, sizeof(Key) * keys.size());
    CudaMemcpy(keys_packets, keys.data(),
    	sizeof(Key) * keys.size(), HostToDevice);
    CudaMalloc((void**)&values_packets, sizeof(Value) * values.size());
    CudaMemcpy(values_packets, values.data(),
    	sizeof(Value) * values.size(), HostToDevice);

	const int threads = THREAD_1D_UNIT;
	const int blocks = UPPER_ALIGN(bucket_count_, THREAD_1D_UNIT);
	InsertHashTableEntriesKernel<<<blocks, threads>>>(
		server_, keys_packets, values_packets, keys.size());
	CudaSynchronize();

	CudaFree(keys_packets);
	CudaFree(values_packets);
}

template<typename Key, typename Value, typename Hasher>
void HashTableCuda<Key, Value, Hasher>
    ::Delete(std::vector<Key> &keys) {

   	Key *keys_packets;
	CudaMalloc((void**)&keys_packets, sizeof(Key) * keys.size());
	CudaMemcpy(keys_packets, keys.data(),
		sizeof(Key) * keys.size(), HostToDevice);

	const int threads = THREAD_1D_UNIT;
	const int blocks = UPPER_ALIGN(bucket_count_, THREAD_1D_UNIT);
	DeleteHashTableEntriesKernel<<<blocks, threads>>>(
		server_, keys_packets, keys.size());
	CudaSynchronize();

	CudaFree(keys_packets);
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
	CudaMalloc((void**)&array_entry_count_packets, sizeof(int) * bucket_count_);
	CudaMalloc((void**)&list_entry_count_packets, sizeof(int) * bucket_count_);

	const int threads = THREAD_1D_UNIT;
	const int blocks = UPPER_ALIGN(bucket_count_, THREAD_1D_UNIT);
	ProfileHashTableKernel<<<blocks, threads>>>(
		server_, array_entry_count_packets, list_entry_count_packets);
	CudaSynchronize();

	CudaMemcpy(array_entry_count.data(), array_entry_count_packets,
		sizeof(int) * bucket_count_, DeviceToHost);
	CudaMemcpy(list_entry_count.data(), list_entry_count_packets,
		sizeof(int) * bucket_count_, DeviceToHost);
	CudaFree(array_entry_count_packets);
	CudaFree(list_entry_count_packets);

	return std::make_tuple(array_entry_count, list_entry_count);
}
}
#endif