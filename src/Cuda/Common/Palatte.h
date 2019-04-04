//
// Created by wei on 4/3/19.
//

#pragma once

#include "Common.h"
#include "LinearAlgebraCuda.h"

namespace open3d {
namespace cuda {
__HOSTDEVICE__
inline Vector3f Jet(float v, float vmin, float vmax) {
    Vector3f c = Vector3f::Ones();

    const float dv = vmax - vmin;
    const float inv_dv = 1.0f / dv;

    v = v < vmin ? vmin : v;
    v = v > vmax ? vmax : v;

    if (v < (vmin + 0.25 * dv)) {
        c(0) = 0;
        c(1) = 4 * (v - vmin) * inv_dv;
    } else if (v < (vmin + 0.5 * dv)) {
        c(0) = 0;
        c(2) = 1 + 4 * (vmin + 0.25 * dv - v) * inv_dv;
    } else if (v < (vmin + 0.75 * dv)) {
        c(0) = 4 * (v - vmin - 0.5 * dv) * inv_dv;
        c(2) = 0;
    } else {
        c(1) = 1 + 4 * (vmin + 0.75 * dv - v) * inv_dv;
        c(2) = 0;
    }

    return (c);
}
}
}