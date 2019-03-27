//
// Created by wei on 1/14/19.
//

#pragma once

#include "Array2DCuda.h"
#include <Cuda/Common/UtilsCuda.h>

namespace open3d {
namespace cuda {

template<typename T>
__device__
T* Array2DCudaDevice<T>::row(int r) {
    return (T *) ((char *) data_ + pitch_ * r);
}

template<typename T>
__device__
int Array2DCudaDevice<T>::expand_rows(int num) {
#ifdef CUDA_DEBUG_ENABLE_ASSERTION
    assert(*iterator_rows_ < max_rows_);
#endif

    int addr = atomicAdd(iterator_rows_, num);
    return addr;
}

template<typename T>
__device__
T& Array2DCudaDevice<T>::at(int r, int c) {
#ifdef CUDA_DEBUG_ENABLE_ASSERTION
    assert(r >= 0 && r < max_rows_);
    assert(c >= 0 && c < max_cols_);
#endif
    T *val = (T *) ((char*) data_ + pitch_ * r) + c;
    return *val;
}

template<typename T>
__device__
T& Array2DCudaDevice<T>::operator() (int r, int c) {
    return at(r, c);
}
}
}