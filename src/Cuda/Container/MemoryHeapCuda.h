/**
 * Created by wei on 18-3-29.
 */

#ifndef _MEMORY_HEAP_CUDA_H_
#define _MEMORY_HEAP_CUDA_H_

#include "Containers.h"
#include <Cuda/Common/Common.h>

#include <vector>

namespace three {
/**
 * Memory allocation and free are expensive on GPU.
 * (And are easy to overflow, I need to check the the reason.)
 * We intend to pre-allocate memory, and manually manage (alloc, free) it.
 * Currently instantiated usages include:
 * - LinkedListNode, in LinkedList<T>
 * - Value, in HashTable<T>
 *
 * Specifically,
 * - LinkedListNode<HashEntry>* node is on MemoryHeap<Node>,
 * - node->data.value_ptr is on MemoryHeap<Value>,
 * it could be confusing, as there are int pointers pointed to int pointers.
 *
 * Basically, we maintain one memory heap per data type.
 * A more general way it is to build an entire memory manager,
 * but that can be too complicated.
 * (And even more complicated when you have have multiple threads!)
 * @TODO: If I have time I will modify CSAPP Malloc Lab's code here.
 */

template<typename T>
class MemoryHeapCuda;

template<typename T>
class MemoryHeapCudaServer {
private:
	T* data_;   /* [N] */
	int* heap_; /* [N] */
	int* heap_counter_; /* [1] */

public:
	int max_capacity_;

public:
	__DEVICE__ int Malloc();
	__DEVICE__ void Free(int addr);

	/* heap (at each index) stores addrs
	 * addrs point to values */
	__DEVICE__ int& get_heap(int index);
	__DEVICE__ T& get_value(int addr);

	friend class MemoryHeapCuda<T>;
};

template<typename T>
class MemoryHeapCuda {
private:
	MemoryHeapCudaServer<T> server_;
	int max_capacity_;
	int HeapCounter();

public:
	MemoryHeapCuda() { max_capacity_ = -1; };
	~MemoryHeapCuda() = default;

	void Init(int max_capacity);
	void Destroy();
	void Reset();

	/* Hopefully this is only used for debugging. */
	std::vector<int> DownloadHeap();
	std::vector<T> DownloadValue();

	int max_capacity() {
		return max_capacity_;
	}
	MemoryHeapCudaServer<T>& server() {
		return server_;
	}
};


template<class T>
__GLOBAL__
void ResetMemoryHeapKernel(MemoryHeapCudaServer<T> server);

};
#endif /* _MEMORY_HEAP_CUDA_H_ */
