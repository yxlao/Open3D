/**
 * Created by wei on 18-3-29.
 */

#pragma once

#include "ContainerClasses.h"
#include <Cuda/Common/Common.h>
#include <memory>

#include <vector>

namespace open3d {
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
    T *data_;   /* [N] */
    int *heap_; /* [N] */
    int *heap_counter_; /* [1] */

public:
    int max_capacity_;

public:
    __HOSTDEVICE__ inline T* data() {
        return data_;
    }
    __HOSTDEVICE__ inline int* heap() {
        return heap_;
    }

    __DEVICE__ int Malloc();
    __DEVICE__ void Free(size_t addr);

    /* heap (at each index) stores addrs
     * addrs point to values */
    __DEVICE__ int &get_heap(size_t index);
    __DEVICE__ T &get_value(size_t addr);
    __DEVICE__ const T& get_value(size_t addr) const;

    friend class MemoryHeapCuda<T>;
};

template<typename T>
class MemoryHeapCuda {
private:
    std::shared_ptr<MemoryHeapCudaServer<T>> server_ = nullptr;
    int max_capacity_;
    int HeapCounter();

public:
    MemoryHeapCuda();
    ~MemoryHeapCuda();
    MemoryHeapCuda(const MemoryHeapCuda<T> &other);
    MemoryHeapCuda<T> &operator=(const MemoryHeapCuda<T> &other);

    void Create(int max_capacity);
    void Release();
    void Reset();

    /* Hopefully this is only used for debugging. */
    std::vector<int> DownloadHeap();
    std::vector<T> DownloadValue();

    int max_capacity() const {
        return max_capacity_;
    }
    std::shared_ptr<MemoryHeapCudaServer<T>> &server() {
        return server_;
    }
    const std::shared_ptr<MemoryHeapCudaServer<T>> &server() const {
        return server_;
    }
};

template<class T>
__GLOBAL__
void ResetMemoryHeapKernel(MemoryHeapCudaServer<T> server);

};