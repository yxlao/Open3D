//
// Created by wei on 18-3-29.
//

#pragma once

#include "MemoryHeapCuda.h"
#include <Cuda/Common/UtilsCuda.h>

#include <Core/Core.h>

#include <cassert>

namespace open3d {

namespace cuda {
/**
 * Server end
 * In fact we don't know which indices are used and which are freed,
 * we can only give a coarse boundary test.
 */
template<typename T>
__device__
int &MemoryHeapCudaDevice<T>::internal_addr_at(size_t index) {
#ifdef CUDA_DEBUG_ENABLE_ASSERTION
    assert(index < max_capacity_);
#endif
    return heap_[index];
}

template<typename T>
__device__
    T
&
MemoryHeapCudaDevice<T>::value_at(size_t
addr) {
#ifdef CUDA_DEBUG_ENABLE_ASSERTION
assert(addr < max_capacity_);
#endif
return data_[addr];
}

template<typename T>
__device__
const T
&
MemoryHeapCudaDevice<T>::value_at(size_t
addr) const {
#ifdef CUDA_DEBUG_ENABLE_ASSERTION
assert(addr < max_capacity_);
#endif
return data_[addr];
}

/**
 * The @value array is FIXED.
 * The @heap array stores the addresses of the values.
 * Only the unallocated part is maintained.
 * (ONLY care about the heap above the heap counter. Below is meaningless.)
 * ---------------------------------------------------------------------
 * heap  ---Malloc-->  heap  ---Malloc-->  heap  ---Free(0)-->  heap
 * N-1                 N-1                  N-1                  N-1   |
 *  .                   .                    .                    .    |
 *  .                   .                    .                    .    |
 *  .                   .                    .                    .    |
 *  3                   3                    3                    3    |
 *  2                   2                    2 <-                 2    |
 *  1                   1 <-                 1                    0 <- |
 *  0 <- heap_counter   0                    0                    0
 */
template<class T>
__device__
int MemoryHeapCudaDevice<T>::Malloc() {
    int index = atomicAdd(heap_counter_, 1);
#ifdef CUDA_DEBUG_ENABLE_ASSERTION
    assert(index < max_capacity_);
#endif
    return heap_[index];
}

template<class T>
__device__
void MemoryHeapCudaDevice<T>::Free(size_t addr) {
    int index = atomicSub(heap_counter_, 1);
#ifdef CUDA_DEBUG_ENABLE_ASSERTION
    assert(index >= 1);
#endif
    heap_[index - 1] = (int) addr;
}
} // cuda
} // open3d