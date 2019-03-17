/**
 * Created by wei on 18-4-9.
 */

#pragma once

#include "ImageCuda.h"
#include "LinearAlgebraCuda.h"
#include <iostream>
#include <driver_types.h>
#include <src/Cuda/Common/UtilsCuda.h>
#include <vector_types.h>

namespace open3d {

namespace cuda {
/**
 * Server end
 */
/**
 * assert will make these functions super slow
 */
template<typename Scalar, size_t Channel>
__device__
VectorCuda<Scalar, Channel>& ImageCudaDevice<Scalar, Channel>::at(
    int x, int y) {
#ifdef CUDA_DEBUG_ENABLE_ASSERTION
    assert(x >= 0 && x < width_);
    assert(y >= 0 && y < height_);
#endif
    VectorCuda<Scalar, Channel> * value = (VectorCuda<Scalar, Channel> *)
        ((char *) data_ + y * pitch_) + x;
    return (*value);
}

template<typename Scalar, size_t Channel>
__device__
Scalar& ImageCudaDevice<Scalar, Channel>::at(int x, int y, int channel) {
#ifdef CUDA_DEBUG_ENABLE_ASSERTION
    assert(x >= 0 && x < width_);
    assert(y >= 0 && y < height_);
    assert(channel >= 0 && channel < Channel);
#endif
    VectorCuda<Scalar, Channel> * value = (VectorCuda<Scalar, Channel> *)
        ((char *) data_ + y * pitch_) + x;
    return (*value)(channel);
}

template<typename Scalar, size_t Channel>
__device__
VectorCuda<Scalar, Channel>& ImageCudaDevice<Scalar, Channel>::operator()
    (int x, int y) {
    return at(x, y);
}

template<typename Scalar, size_t Channel>
__device__
Scalar& ImageCudaDevice<Scalar, Channel>::operator()
    (int x, int y, int channel) {
    return at(x, y, channel);
}

/**
 * Naive interpolation without considering holes in depth images.
 */
template<typename Scalar, size_t Channel>
__device__
VectorCuda<Scalar, Channel> ImageCudaDevice<Scalar, Channel>::interp_at(float x, float y) {
#ifdef CUDA_DEBUG_ENABLE_ASSERTION
    assert(x >= 0 && x < width_ - 1);
    assert(y >= 0 && y < height_ - 1);
#endif
    int x0 = (int) floor(x), y0 = (int) floor(y);
    float a = x - x0, b = y - y0;

    VectorCuda<Scalar, Channel> v00, v01, v11, v10, ret;
    v00 = at(x0, y0);
    v01 = at(x0, y0 + 1);
    v11 = at(x0 + 1, y0 + 1);
    v10 = at(x0 + 1, y0);

    float w00, w01, w11, w10;
    w00 = (1 - a) * (1 - b);
    w01 = (1 - a) * b;
    w11 = a * b;
    w10 = a * (1 - b);

#pragma unroll 1
    for (int i = 0; i < Channel; ++i) {
        ret(i) = w00 * v00(i) + w01 * v01(i) + w11 * v11(i) + w10 * v10(i);
    }

    return ret;
}

template<typename Scalar, size_t Channel>
__device__
VectorCuda<Scalar, Channel>
ImageCudaDevice<Scalar, Channel>::BoxFilter2x2(int x, int y) {
#ifdef CUDA_DEBUG_ENABLE_ASSERTION
    assert(x >= 0 && x < width_);
    assert(y >= 0 && y < height_);
#endif

    int xp1 = min(width_ - 1, x + 1);
    int yp1 = min(height_ - 1, y + 1);

    VectorCuda<Scalar, Channel> v00, v01, v11, v10, ret(0);
    v00 = at(x, y);
    v01 = at(x, yp1);
    v11 = at(xp1, yp1);
    v10 = at(xp1, y);

#pragma unroll 1
    for (int i = 0; i < Channel; ++i) {
        ret(i) = 0.25f * (v00(i) + v01(i) + v11(i) + v10(i));
    }

    return ret;
}

template<typename Scalar, size_t Channel>
__device__
VectorCuda<Scalar, Channel>
ImageCudaDevice<Scalar, Channel>::GaussianFilter(int x, int y, int kernel_idx) {
#ifdef CUDA_DEBUG_ENABLE_ASSERTION
    assert(x >= 0 && x < width_);
    assert(y >= 0 && y < height_);
#endif

    static const int kernel_sizes[3] = {3, 5, 7};
    static const float gaussian_weights[3][7] = {
        {0.25f, 0.5f, 0.25f, 0, 0, 0, 0},
        {0.0625f, 0.25f, 0.375f, 0.25f, 0.0625f, 0, 0},
        {0.03125f, 0.109375f, 0.21875f, 0.28125f, 0.21875f, 0.109375f, 0.03125f}
    };

    const int kernel_size = kernel_sizes[kernel_idx];
    const int kernel_size_2 = kernel_size >> 1;
    const float *kernel = gaussian_weights[kernel_idx];

    int x_min = max(0, x - kernel_size_2);
    int y_min = max(0, y - kernel_size_2);
    int x_max = min(width_ - 1, x + kernel_size_2);
    int y_max = min(height_ - 1, y + kernel_size_2);

    auto sum_val = VectorCuda<float, Channel>::Zeros();
    float sum_weight = 0;

    for (int xx = x_min; xx <= x_max; ++xx) {
        for (int yy = y_min; yy <= y_max; ++yy) {
            auto val = at(xx, yy).template cast<float>();
            float weight = kernel[xx - x + kernel_size_2]
                * kernel[yy - y + kernel_size_2];
            sum_val += val * weight;
            sum_weight += weight;
        }
    }

    sum_val /= sum_weight;

    return sum_val.template cast<Scalar>();
}

template<typename Scalar, size_t Channel>
__device__
VectorCuda<Scalar, Channel>
ImageCudaDevice<Scalar, Channel>::BilateralFilter(
    int x, int y, int kernel_idx, float val_sigma) {
#ifdef CUDA_DEBUG_ENABLE_ASSERTION
    assert(x >= 0 && x < width_);
    assert(y >= 0 && y < height_);
#endif

    static const int kernel_sizes[3] = {3, 5, 7};
    static const float gaussian_weights[3][7] = {
        {0.25f, 0.5f, 0.25f, 0, 0, 0, 0},
        {0.0625f, 0.25f, 0.375f, 0.25f, 0.0625f, 0, 0},
        {0.03125f, 0.109375f, 0.21875f, 0.28125f, 0.21875f, 0.109375f, 0.03125f}
    };

    const int kernel_size = kernel_sizes[kernel_idx];
    const int kernel_size_2 = kernel_size >> 1;
    const float *kernel = gaussian_weights[kernel_idx];

    int x_min = max(0, x - kernel_size_2);
    int y_min = max(0, y - kernel_size_2);
    int x_max = min(width_ - 1, x + kernel_size_2);
    int y_max = min(height_ - 1, y + kernel_size_2);

    auto center_val = at(x, y).template cast<float>();
    auto sum_val = VectorCuda<float, Channel>::Zeros();
    float sum_weight = 0;

    for (int xx = x_min; xx <= x_max; ++xx) {
        for (int yy = y_min; yy <= y_max; ++yy) {
            auto val = at(xx, yy).template cast<float>();
            float weight = kernel[xx - x + kernel_size_2]
                * kernel[yy - y + kernel_size_2];
            float value_diff = (val - center_val).norm() / val_sigma;
            weight *= expf(-value_diff);

            sum_val += val * weight;
            sum_weight += weight;
        }
    }

    sum_val /= sum_weight;
    return sum_val.template cast<Scalar>();
}

template<typename Scalar, size_t Channel>
__device__
ImageCudaDevice<Scalar, Channel>::Grad
ImageCudaDevice<Scalar, Channel>::Sobel(int x, int y) {
#ifdef CUDA_DEBUG_ENABLE_ASSERTION
    assert(x >= 1 && x < width_ - 1);
    assert(y >= 1 && y < height_ - 1);
#endif

    int xm1 = max(x - 1, 0), ym1 = max(y - 1, 0);
    int xp1 = min(x + 1, width_ - 1), yp1 = min(y + 1, height_ - 1);
    auto Iumvm = at(xm1, ym1);
    auto Iumv0 = at(xm1, y);
    auto Iumvp = at(xm1, yp1);
    auto Iu0vm = at(x, ym1);
    auto Iu0vp = at(x, yp1);
    auto Iupvm = at(xp1, ym1);
    auto Iupv0 = at(xp1, y);
    auto Iupvp = at(xp1, yp1);

    VectorCuda<float, Channel> dx, dy;

#pragma unroll 1
    for (int i = 0; i < Channel; ++i) {
        dx(i) = (Iupvm(i) - Iumvm(i))
              + (Iupv0(i) - Iumv0(i)) * 2
              + (Iupvp(i) - Iumvp(i));
        dy(i) = (Iumvp(i) - Iumvm(i))
              + (Iu0vp(i) - Iu0vm(i)) * 2
              + (Iupvp(i) - Iupvm(i));
    }

    return {dx, dy};
}
} // cuda
} // open3d