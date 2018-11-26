//
// Created by wei on 18-9-24.
//

#include "MemoryHeapCudaDevice.cuh"

namespace open3d {

template<typename T>
__global__
void ResetMemoryHeapKernel(MemoryHeapCudaServer<T> server) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < server.max_capacity_) {
        server.value_at(i) = T(); /* This is not necessary. */
        server.internal_addr_at(i) = i;
    }
}

template<typename T>
void MemoryHeapCudaKernelCaller<T>::ResetMemoryHeapKernelCaller(
    MemoryHeapCudaServer<T> &server, int max_capacity) {
    const int blocks = DIV_CEILING(max_capacity, THREAD_1D_UNIT);
    const int threads = THREAD_1D_UNIT;

    ResetMemoryHeapKernel << < blocks, threads >> > (server);
    CheckCuda(cudaDeviceSynchronize());
    CheckCuda(cudaGetLastError());
}

template<typename T>
__global__
void ResizeMemoryHeapKernel(MemoryHeapCudaServer<T> src, /* old size */
                            MemoryHeapCudaServer<T> dst  /* new size */) {
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
void MemoryHeapCudaKernelCaller<T>::ResizeMemoryHeapKernelCaller(
    MemoryHeapCudaServer<T> &server, MemoryHeapCudaServer<T> &dst,
    int new_max_capacity) {
    const int blocks = DIV_CEILING(new_max_capacity, THREAD_1D_UNIT);
    const int threads = THREAD_1D_UNIT;

    ResizeMemoryHeapKernel << < blocks, threads >> > (server, dst);
    CheckCuda(cudaDeviceSynchronize());
    CheckCuda(cudaGetLastError());
}
}