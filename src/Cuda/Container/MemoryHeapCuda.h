/**
 * Created by wei on 18-3-29.
 */

#pragma once

#include "ContainerClasses.h"

#include <Cuda/Common/Common.h>

#include <memory>
#include <vector>

namespace open3d {

namespace cuda {
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
class MemoryHeapCudaDevice {
private:
    T *data_;   /* [N] */
    int *heap_; /* [N] */
    int *heap_counter_; /* [1] */

public:
    int max_capacity_;

public:
    __HOSTDEVICE__ inline T *data() {
        return data_;
    }
    __HOSTDEVICE__ inline int *heap() {
        return heap_;
    }
    __HOSTDEVICE__ inline int *heap_counter() {
        return heap_counter_;
    }

    __DEVICE__ int Malloc();
    __DEVICE__ void Free(size_t addr);

    __DEVICE__ int &internal_addr_at(size_t index);
    __DEVICE__ T &value_at(size_t addr);
    __DEVICE__ const T &value_at(size_t addr) const;

    friend class MemoryHeapCuda<T>;
};

template<typename T>
class MemoryHeapCuda {
public:
    std::shared_ptr<MemoryHeapCudaDevice<T>> device_ = nullptr;
    int HeapCounter();

public:
    int max_capacity_;

public:
    MemoryHeapCuda();
    ~MemoryHeapCuda();
    MemoryHeapCuda(const MemoryHeapCuda<T> &other);
    MemoryHeapCuda<T> &operator=(const MemoryHeapCuda<T> &other);

    void Resize(int max_capacity);
    void Create(int max_capacity);
    void Release();
    void Reset();

    /* Hopefully this is only used for debugging. */
    std::vector<int> DownloadHeap();
    std::vector<T> DownloadValue();
};

template<typename T>
class MemoryHeapCudaKernelCaller {
public:
    static __HOST__ void ResetMemoryHeapKernelCaller(
        MemoryHeapCudaDevice<T> &server, int max_capacity);
    static __HOST__ void ResizeMemoryHeapKernelCaller(
        MemoryHeapCudaDevice<T> &server,
        MemoryHeapCudaDevice<T> &dst,
        int new_max_capacity);
};

template<class T>
__GLOBAL__
void ResetMemoryHeapKernel(MemoryHeapCudaDevice<T> server);

template<typename T>
__GLOBAL__
void ResizeMemoryHeapKernel(MemoryHeapCudaDevice<T> src,
                            MemoryHeapCudaDevice<T> dst);

} // cuda
} // open3d