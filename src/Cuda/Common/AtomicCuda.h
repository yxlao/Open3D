//
// Created by wei on 11/2/18.
//

#pragma once

#include "UtilsCuda.h"

namespace open3d {

#ifdef __CUDACC__
__device__
inline float atomicMinf(float *addr, float value) {
    float old;
    old = (value >= 0) ?
        __int_as_float(atomicMin((int *) addr, __float_as_int(value))) :
        __uint_as_float(atomicMax((unsigned int *) addr, __float_as_uint(value)));
    return old;
}

__device__
inline float atomicMaxf(float *addr, float value) {
    float old;
    old = (value >= 0) ?
        __int_as_float(atomicMax((int *) addr, __float_as_int(value))) :
        __uint_as_float(atomicMin((unsigned int *) addr, __float_as_uint(value)));
    return old;
}
#endif

}