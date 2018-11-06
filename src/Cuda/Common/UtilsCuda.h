//
// Created by wei on 9/27/18.
//

#ifndef OPEN3D_UTILSCUDA_H
#define OPEN3D_UTILSCUDA_H

#include <cstdlib>
#include <driver_types.h>
#include <cuda_runtime_api.h>

#include "Common.h"
#include "HelperCuda.h"

namespace open3d {

/** If this is on, perform boundary checks! **/
#define CUDA_DEBUG_ENABLE_ASSERTION_
#define CUDA_DEBUG_ENABLE_PRINTF_
#define HOST_DEBUG_MONITOR_LIFECYCLE_

#define CheckCuda(val)  check ( (val), #val, __FILE__, __LINE__ )

}

#endif //OPEN3D_UTILSCUDA_H
