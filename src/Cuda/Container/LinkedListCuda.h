/**
 * Created by wei on 18-4-2.
 */

#pragma once

#include "ContainerClasses.h"
#include "MemoryHeapCuda.h"

#include <Cuda/Common/Common.h>

namespace open3d {

namespace cuda {
/**
 * It is a basic unit, so we DON'T wrap it with device codes
 * otherwise pointer management will make you cry.
 */
template<typename T>
struct LinkedListNodeCuda {
    int next_node_ptr = 0;
    T data;
};

/**
 * This class is NOT thread safe, because comparing to Array, its operation
 * is too complicated and cannot be finished using only 'atomicAdd'.
 * Manually add LOCKs per linked list if you want to read / write in parallel.
 */
template<typename T>
class LinkedListCudaDevice {
private:
    typedef MemoryHeapCudaDevice<LinkedListNodeCuda<T>> MemoryHeapDevice;
    int *head_node_ptr_;
    int *size_;

    MemoryHeapDevice memory_heap_;

public:
    int max_capacity_;

public:
    /**
     * WARNING!!! USE ONLY ONE VERSION OF Create AND Release!!!
     * As a generic container, you can instantiate the class on CPU, and call
     * device functions.
     * For our HashTable, we have to instantiate the class ON GPU, therefore
     * we need a GPU version.
     * Choose the correct version of Create and Release depending on where you
     * instantiate it.
     */
    __DEVICE__ void Create(MemoryHeapDevice &memory_heap_device,
                           int *head_node_ptr, int *size_ptr);
    __DEVICE__ void Release();

    __DEVICE__ void Insert(T value);
    __DEVICE__ int Delete(int node_ptr);
    __DEVICE__ int Find(T value) const;
    __DEVICE__ int FindAndDelete(T value);
    __DEVICE__ void Clear();

    __DEVICE__ inline int head_node_ptr() {
        return *head_node_ptr_;
    }
    __DEVICE__ inline int size() {
        return *size_;
    }
    __DEVICE__
    LinkedListNodeCuda<T> &get_node(int node_ptr) {
        return memory_heap_.value_at(node_ptr);
    }
    __DEVICE__
    const LinkedListNodeCuda<T> &get_node(int node_ptr) const {
        return memory_heap_.value_at(node_ptr);
    }

    friend class LinkedListCuda<T>;
};

template<typename T>
class LinkedListCuda {
public:
    typedef MemoryHeapCuda<LinkedListNodeCuda<T>> MemoryHeap;

    std::shared_ptr<LinkedListCudaDevice<T>> device_ = nullptr;
    MemoryHeap memory_heap_;

public:
    int max_capacity_;

public:
    LinkedListCuda();
    LinkedListCuda(const LinkedListCuda<T> &other);
    LinkedListCuda<T> &operator=(const LinkedListCuda<T> &other);
    ~LinkedListCuda();

    void Create(int max_capacity, MemoryHeap &memory_heap);
    void Release();

    /* Mainly for test usages. Only launched with 1 thread. */
    void Insert(std::vector<int> &data);
    void Find(std::vector<int> &data);
    void Delete(std::vector<int> &data);
    std::vector<T> Download();

    void UpdateDevice();

    int size();

    MemoryHeap &memory_heap() {
        return memory_heap_;
    }
    const MemoryHeap &memory_heap() const {
        return memory_heap_;
    }
};

template<typename T>
class LinkedListCudaKernelCaller {
public:
    static void Insert(LinkedListCuda<T> &device, ArrayCuda<T> &data);
    static void Find(LinkedListCuda<T> &device, ArrayCuda<T> &query);
    static void Delete(LinkedListCuda<T> &device, ArrayCuda<T> &query);
    static void Clear(LinkedListCuda<T> &device);
    static void Download(LinkedListCuda<T> &device, ArrayCuda<T> &data);
};

template<typename T>
__GLOBAL__
void InsertKernel(LinkedListCudaDevice<T> device, ArrayCudaDevice<T> data);
template<typename T>
__GLOBAL__
void FindKernel(LinkedListCudaDevice<T> device, ArrayCudaDevice<T> query);
template<typename T>
__GLOBAL__
void DeleteKernel(LinkedListCudaDevice<T> device, ArrayCudaDevice<T> query);
template<typename T>
__GLOBAL__
void ClearKernel(LinkedListCudaDevice<T> device);
template<typename T>
__GLOBAL__
void DownloadKernel(LinkedListCudaDevice<T> device, ArrayCudaDevice<T> data);
} // cuda
} // open3d