//
// Created by wei on 9/27/18.
//

#pragma once

#include <cstdlib>

#include <Open3D/Utility/Console.h>
#include <Open3D/Utility/Timer.h>

#include "driver_types.h"
#include "cuda_runtime_api.h"

#include "Common.h"
#include "HelperCuda.h"

namespace open3d {

namespace cuda {
/** If this is on, perform boundary checks! **/
#define CUDA_DEBUG_ENABLE_ASSERTION_
#define CUDA_DEBUG_ENABLE_PRINTF_
#define HOST_DEBUG_MONITOR_LIFECYCLE_

#define CheckCuda(val)  check ( (val), #val, __FILE__, __LINE__ )

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

} // cuda
} // open3d