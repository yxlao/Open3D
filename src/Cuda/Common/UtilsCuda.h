//
// Created by wei on 9/27/18.
//

#ifndef OPEN3D_UTILSCUDA_H
#define OPEN3D_UTILSCUDA_H

#include "Common.h"
#include "HelperCuda.h"
#include <cstdlib>
#include <driver_types.h>

namespace three {

#define CheckCuda(val)  check ( (val), #val, __FILE__, __LINE__ )

}

#endif //OPEN3D_UTILSCUDA_H
