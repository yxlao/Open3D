/**
 * Created by wei on 18-4-2.
 */

#ifndef _LINKED_LIST_CUDA_CUH_
#define _LINKED_LIST_CUDA_CUH_

#include "LinkedListCuda.h"
#include "MemoryHeapCuda.cuh"

#include <Cuda/Common/UtilsCuda.h>

#include <cstdio>
#include <cassert>

namespace open3d {
/**
 * Server end
 */
template<typename T>
__device__
void LinkedListCudaServer<T>::Clear() {
	int node_ptr = *head_node_ptr_;
	while (node_ptr != NULL_PTR) {
		int next_node_ptr = memory_heap_.get_value(node_ptr).next_node_ptr;
		memory_heap_.Free(node_ptr);
		node_ptr = next_node_ptr;
		(*size_)--;
	}
	*head_node_ptr_ = NULL_PTR;
	assert((*size_) == 0);
}

template<typename T>
__device__
void LinkedListCudaServer<T>::Insert(T value) {
	int node_ptr = memory_heap_.Malloc();
	if ((*size_) >= max_capacity_) {
		printf("Linked list full!\n");
		return;
	}

	memory_heap_.get_value(node_ptr).data = value;
	memory_heap_.get_value(node_ptr).next_node_ptr = *head_node_ptr_;
	*head_node_ptr_ = node_ptr;

	(*size_)++;
}

/**
 * Cuda version of Createialization.
 * It is more intuitive to directly allocate it in kernel with malloc(),
 * but it fails with not too many number of operations
 * @TODO check why.
 * A way around is to pre-allocate it on the host side using CheckCuda(cudaMalloc(),
 * and assign these space directly to the pointers.
 * @tparam T
 * @param memory_heap_server
 * @param head_node_ptr
 * @param size
 * @return
 */
template<typename T>
__device__
void LinkedListCudaServer<T>::Create(
	LinkedListCudaServer<T>::MemoryHeapLinkedListNodeCudaServer &memory_heap_server,
	int* head_node_ptr,
	int* size_ptr) {
	max_capacity_ = memory_heap_server.max_capacity_;
	memory_heap_ = memory_heap_server;
	head_node_ptr_ = head_node_ptr;
	size_ = size_ptr;
	(*head_node_ptr_) = NULL_PTR;
	(*size_) = 0;
}

/* Release the assigned pointers externally */
template<typename T>
__device__
void LinkedListCudaServer<T>::Release() {}

template<typename T>
__device__
int LinkedListCudaServer<T>::Delete(const int node_ptr) {
	if (*head_node_ptr_ == NULL_PTR || node_ptr == NULL_PTR) {
#ifdef __CUDAPRINT__
		printf("Error: Invalid pointer or linked list!\n");
#endif
		return NODE_NOT_FOUND;
	}

	/* 1. Head */
	if (*head_node_ptr_ == node_ptr) {
		*head_node_ptr_ = memory_heap_.get_value(*head_node_ptr_).next_node_ptr;
		memory_heap_.Free(node_ptr);
		(*size_)--;
		return SUCCESS;
	}

	/* 2. Search in the linked list for its predecessor */
	int node_ptr_pred = *head_node_ptr_;
	while (memory_heap_.get_value(node_ptr_pred).next_node_ptr != node_ptr
		&& memory_heap_.get_value(node_ptr_pred).next_node_ptr != NULL_PTR) {
		node_ptr_pred = memory_heap_.get_value(node_ptr_pred).next_node_ptr;
	}

	if (memory_heap_.get_value(node_ptr_pred).next_node_ptr == NULL_PTR) {
#ifdef __CUDAPRINT__
		printf("Error: Node_ptr %d not found!\n", node_ptr);
#endif
		return NODE_NOT_FOUND;
	}

	memory_heap_.get_value(node_ptr_pred).next_node_ptr =
		memory_heap_.get_value(node_ptr).next_node_ptr;
	memory_heap_.Free(node_ptr);
	(*size_)--;
	return SUCCESS;
}

template<typename T>
__device__
int LinkedListCudaServer<T>::Find(T value) {
	int node_ptr = *head_node_ptr_;
	while (node_ptr != NULL_PTR) {
		if (memory_heap_.get_value(node_ptr).data == value)
			return node_ptr;
		node_ptr = memory_heap_.get_value(node_ptr).next_node_ptr;
	}

#ifdef __CUDAPRINT__
	printf("Error: Value not found!\n");
#endif
	return NODE_NOT_FOUND;
}

template<class T>
__device__
int LinkedListCudaServer<T>::FindAndDelete(T value) {
	if (*head_node_ptr_ == NULL_PTR) {
#ifdef __CUDAPRINT__
		printf("Empty linked list!\n");
#endif
		return NODE_NOT_FOUND;
	}

	/* 1. Head */
	if (memory_heap_.get_value(*head_node_ptr_).data == value) {
		/* NULL_PTR and ! NULL_PTR, both cases are ok */
		int next_node_ptr = memory_heap_.get_value(*head_node_ptr_)
		.next_node_ptr;
		memory_heap_.Free(*head_node_ptr_);
		*head_node_ptr_ = next_node_ptr;
		(*size_)--;
		return SUCCESS;
	}

	/* 2. Search in the linked list for its predecessor */
	int node_ptr_pred = *head_node_ptr_;
	int node_ptr_curr = memory_heap_.get_value(*head_node_ptr_).next_node_ptr;
	while (node_ptr_curr != NULL_PTR) {
		T& data = memory_heap_.get_value(node_ptr_curr).data;
		if (data == value) break;
		node_ptr_pred = node_ptr_curr;
		node_ptr_curr = memory_heap_.get_value(node_ptr_curr).next_node_ptr;
	}

	if (node_ptr_curr == NULL_PTR) {
#ifdef __CUDAPRINT__
		printf("Error: Value not found!\n");
#endif
		return NODE_NOT_FOUND;
	}

	memory_heap_.get_value(node_ptr_pred).next_node_ptr =
		memory_heap_.get_value(node_ptr_curr).next_node_ptr;
	memory_heap_.Free(node_ptr_curr);
	(*size_)--;
	return SUCCESS;
}

/**
 * Client end
 */
template<typename T>
void LinkedListCuda<T>::Create(int max_capacity,
	MemoryHeapCuda<LinkedListNodeCuda<T>> &memory_heap) {
	assert(max_capacity < memory_heap.server().max_capacity_);

	max_capacity_ = max_capacity;
	server_.max_capacity_ = max_capacity;

	server_.memory_heap_ = memory_heap.server(); /* Copy constructor! */
	CheckCuda(cudaMalloc((void**)&server_.head_node_ptr_, sizeof(int)));
	CheckCuda(cudaMemset(server_.head_node_ptr_, NULL_PTR, sizeof(int)));

	CheckCuda(cudaMalloc((void**)&server_.size_, sizeof(int)));
	CheckCuda(cudaMemset(server_.size_, 0, sizeof(int)));
}

template<typename T>
void LinkedListCuda<T>::Release() {
	ClearLinkedListKernel<<<1, 1>>>(server_);
	CheckCuda(cudaDeviceSynchronize());
}

template<typename T>
int LinkedListCuda<T>::size() {
	int ret;
	CheckCuda(cudaMemcpy(&ret, server_.size_, sizeof(int), cudaMemcpyDeviceToHost));
	return ret;
}

template<typename T>
void LinkedListCuda<T>::Insert(std::vector<int> &data) {
	T* data_packages;
	CheckCuda(cudaMalloc((void**)&data_packages, sizeof(T) * data.size()));
	CheckCuda(cudaMemcpy(data_packages, data.data(),
		sizeof(T) * data.size(), cudaMemcpyHostToDevice));

	InsertLinkedListKernel<<<1, 1>>>(
		server_, data_packages, data.size());
	CheckCuda(cudaDeviceSynchronize());

	CheckCuda(cudaFree(data_packages));
}

template<typename T>
void LinkedListCuda<T>::Find(std::vector<int> &query) {
	T* data_packages;
	CheckCuda(cudaMalloc((void**)&data_packages, sizeof(T) * query.size()));
	CheckCuda(cudaMemcpy(data_packages, query.data(),
		sizeof(T) * query.size(), cudaMemcpyHostToDevice));

	FindLinkedListKernel<<<1, 1>>>(
		server_, data_packages, query.size());
	CheckCuda(cudaDeviceSynchronize());

	CheckCuda(cudaFree(data_packages));
}

template<typename T>
void LinkedListCuda<T>::Delete(std::vector<int> &query) {
	T* data_packages;
	CheckCuda(cudaMalloc((void**)&data_packages, sizeof(T) * query.size()));
	CheckCuda(cudaMemcpy(data_packages, query.data(),
		sizeof(T) * query.size(), cudaMemcpyHostToDevice));

	DeleteLinkedListKernel<<<1, 1>>>(
		server_, data_packages, query.size());
	CheckCuda(cudaDeviceSynchronize());

	CheckCuda(cudaFree(data_packages));
}

template<typename T>
std::vector<T> LinkedListCuda<T>::Download() {
	std::vector<T> ret;
	int linked_list_size = size();
	if (linked_list_size == 0) return ret;

	ret.resize(linked_list_size);

	T* data_packages;
	CheckCuda(cudaMalloc((void**)&data_packages, sizeof(T) * linked_list_size));

	DownloadLinkedListKernel<<<1, 1>>>(
		server_, data_packages, linked_list_size);
	CheckCuda(cudaDeviceSynchronize());

	CheckCuda(cudaMemcpy(ret.data(), data_packages,
		sizeof(T) * linked_list_size, cudaMemcpyDeviceToHost));
	CheckCuda(cudaFree(data_packages));

	return ret;
}
};
#endif