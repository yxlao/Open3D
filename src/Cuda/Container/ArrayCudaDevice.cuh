/**
 * Created by wei on 18-4-2.
 */

#pragma once

#include "ArrayCuda.h"
#include <Cuda/Common/UtilsCuda.h>

#include <cassert>

namespace open3d {

namespace cuda {
/**
 * Server end
 */
template<typename T>
__device__
inline int ArrayCudaServer<T>::push_back(T value) {
#ifdef CUDA_DEBUG_ENABLE_ASSERTION
    assert(*iterator_ < max_capacity_);
#endif

    int addr = atomicAdd(iterator_, 1);
    data_[addr] = value;
    return addr;
}

template<typename T>
__device__
inline T &ArrayCudaServer<T>::at(size_t index) {
#ifdef CUDA_DEBUG_ENABLE_ASSERTION
    assert(index <= max_capacity_);
#endif

    return data_[index];
}

template<typename T>
__device__
inline T &ArrayCudaServer<T>::operator[](size_t index) {
#ifdef CUDA_DEBUG_ENABLE_ASSERTION
    assert(index <= max_capacity_);
#endif

    return data_[index];
}
} // cuda
} // open3d