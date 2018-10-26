//
// Created by wei on 10/3/18.
//

#pragma once

#include "VectorCuda.h"

namespace open3d {

/**
 * This class should be passed to server, but as it is a storage class, we
 * don't name it as xxxServer
 */
template<size_t N>
class PinholeCameraCuda {
public:
    int width_[N];
    int height_[N];

    float inv_fx_[N];
    float inv_fy_[N];
    float fx_[N];
    float fy_[N];
    float cx_[N];
    float cy_[N];

public:
    __HOSTDEVICE__ inline int width(size_t level = 0) { return width_[level]; }
    __HOSTDEVICE__ inline int height(size_t level = 0) { return height_[level]; }
    __HOSTDEVICE__ inline float fx(size_t level = 0) { return fx_[level]; }
    __HOSTDEVICE__ inline float fy(size_t level = 0) { return fy_[level]; }
    __HOSTDEVICE__ inline float cx(size_t level = 0) { return cx_[level]; }
    __HOSTDEVICE__ inline float cy(size_t level = 0) { return cy_[level]; }

    __HOSTDEVICE__ void SetUp(
        int in_width = 640, int in_height = 480,
        float in_fx = 525.0f, float in_fy = 525.0f,
        float in_cx = 319.5f, float in_cy = 239.5f) {
        for (int i = 0; i < N; ++i) {

            float factor = 1.0f / (1 << i);
            width_[i] = in_width >> i;
            height_[i] = in_height >> i;
            fx_[i] = in_fx * factor;
            inv_fx_[i] = 1.0f / fx_[i];
            fy_[i] = in_fy * factor;
            inv_fy_[i] = 1.0f / fy_[i];
            cx_[i] = in_cx * factor;
            cy_[i] = in_cy * factor;
        }
    }

    __HOSTDEVICE__ inline bool IsValid(const Vector2f &p, size_t level = 0) {
#ifdef CUDA_DEBUG_ENABLE_ASSERTION
        assert(level < N);
#endif
        return p(0) >= 0 && p(0) < width_[level] - 1
            && p(1) >= 0 && p(1) < height_[level] - 1;
    }

    __HOSTDEVICE__ bool IsInFrustum(const Vector3f &X, size_t level = 0) {
#ifdef CUDA_DEBUG_ENABLE_ASSERTION
        assert(level < N);
#endif
        /* TODO: Derive a RGBDImage Class (using short),
         * holding depth constraints */
        if (X(2) < 0.1 || X(2) > 3) return false;
        return IsValid(Projection(X, level), level);
    }

    __HOSTDEVICE__ Vector2f Projection(const Vector3f &X, size_t level = 0) {
#ifdef CUDA_DEBUG_ENABLE_ASSERTION
        assert(level < N);
#endif
        return Vector2f((fx_[level] * X(0)) / X(2) + cx_[level],
                        (fy_[level] * X(1)) / X(2) + cy_[level]);
    }

    __HOSTDEVICE__ Vector3f InverseProjection(const Vector2f &p, float d,
                                              size_t level = 0) {
#ifdef CUDA_DEBUG_ENABLE_ASSERTION
        assert(level < N);
#endif
        return Vector3f(d * (p(0) - cx_[level]) * inv_fx_[level],
                        d * (p(1) - cy_[level]) * inv_fy_[level],
                        d);
    }

    __HOSTDEVICE__ Vector3f InverseProjection(const Vector2i &p, float d,
                                              size_t level = 0) {
#ifdef CUDA_DEBUG_ENABLE_ASSERTION
        assert(level < N);
#endif
        return Vector3f(d * (p(0) - cx_[level]) * inv_fx_[level],
                        d * (p(1) - cy_[level]) * inv_fy_[level],
                        d);
    }
};

typedef PinholeCameraCuda<1> MonoPinholeCameraCuda;
}
