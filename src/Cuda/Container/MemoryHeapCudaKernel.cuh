//
// Created by wei on 18-9-24.
//

#include "MemoryHeapCudaDevice.cuh"

namespace open3d {
namespace cuda {
template<typename T>
__global__
void ResetMemoryHeapKernel(MemoryHeapCudaDevice<T> device) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < device.max_capacity_) {
        device.value_at(i) = T(); /* This is not necessary. */
        device.internal_addr_at(i) = i;
    }
}

template<typename T>
void MemoryHeapCudaKernelCaller<T>::Reset(MemoryHeapCuda<T> &memory_heap) {
    const int blocks = DIV_CEILING(memory_heap.max_capacity_, THREAD_1D_UNIT);
    const int threads = THREAD_1D_UNIT;

    ResetMemoryHeapKernel << < blocks, threads >> > (*memory_heap.device_);
    CheckCuda(cudaDeviceSynchronize());
    CheckCuda(cudaGetLastError());
}

template<typename T>
__global__
void ResizeMemoryHeapKernel(MemoryHeapCudaDevice<T> src, /* old size */
                            MemoryHeapCudaDevice<T> dst  /* new size */) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i == 0) {
        *dst.heap_counter() = *src.heap_counter();
    }
    if (i < src.max_capacity_) {
        dst.value_at(i) = src.value_at(i);
        dst.internal_addr_at(i) = src.internal_addr_at(i);
    } else if (i < dst.max_capacity_) {
        dst.value_at(i) = T();
        dst.internal_addr_at(i) = i;
    }
}

template<typename T>
void MemoryHeapCudaKernelCaller<T>::Resize(MemoryHeapCuda<T> &src,
                                           MemoryHeapCuda<T> &dst) {
    const int blocks = DIV_CEILING(dst.max_capacity_, THREAD_1D_UNIT);
    const int threads = THREAD_1D_UNIT;

    ResizeMemoryHeapKernel << < blocks, threads >> >(
        *src.device_, *dst.device_);
    CheckCuda(cudaDeviceSynchronize());
    CheckCuda(cudaGetLastError());
}
} // cuda
} // open3d