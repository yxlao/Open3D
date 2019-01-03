/**
 * Created by wei on 18-4-9.
 */

#pragma once

#include "ImageCuda.h"
#include "VectorCuda.h"
#include <iostream>
#include <driver_types.h>
#include <Cuda/Common/UtilsCuda.h>
#include <vector_types.h>
#include <Core/Core.h>

namespace open3d {

namespace cuda {
/**
 * Server end
 */
/**
 * assert will make these functions super slow
 */
template<typename VecType>
__device__
    VecType
&
ImageCudaDevice<VecType>::at(int x, int y) {
#ifdef CUDA_DEBUG_ENABLE_ASSERTION
    assert(x >= 0 && x < width_);
    assert(y >= 0 && y < height_);
#endif
    VecType * value = (VecType *) ((char *) data_ + y * pitch_) + x;
    return (*value);
}

template<typename VecType>
__device__
    VecType
&
ImageCudaDevice<VecType>::operator()(int x, int y) {
    return at(x, y);
}

/**
 * Naive interpolation without considering holes in depth images.
 */
template<typename VecType>
__device__
    VecType
ImageCudaDevice<VecType>::interp_at(float x, float y) {
#ifdef CUDA_DEBUG_ENABLE_ASSERTION
    assert(x >= 0 && x < width_ - 1);
    assert(y >= 0 && y < height_ - 1);
#endif
    int x0 = (int) floor(x), y0 = (int) floor(y);
    float a = x - x0, b = y - y0;
    return VecType::FromVectorf(
        (1 - a) * (1 - b) * at(x0, y0).ToVectorf()
            + (1 - a) * b * at(x0, y0 + 1).ToVectorf()
            + a * b * at(x0 + 1, y0 + 1).ToVectorf()
            + a * (1 - b) * at(x0 + 1, y0).ToVectorf());
}

/** SO many if. If it is slow, fall back to get **/
template<typename VecType>
__device__
    VecType
ImageCudaDevice<VecType>::interp_with_holes_at(float x, float y) {
#ifdef CUDA_DEBUG_ENABLE_ASSERTION
    assert(x >= 0 && x < width_ - 1);
    assert(y >= 0 && y < height_ - 1);
#endif

    int x0 = (int) floor(x), y0 = (int) floor(y);
    float a = x - x0, b = y - y0;

    float sum_w = 0;
    auto sum_val = VecType::VecTypef::Zeros();
    auto zero = VecType::VecTypef(0);

    auto val = at(x0, y0).ToVectorf();
    float w = (1 - a) * (1 - b);
    sum_val += w * val;
    sum_w += (val == zero) ? 0 : w;

    val = at(x0, y0 + 1).ToVectorf();
    w = (1 - a) * b;
    sum_val += w * val;
    sum_w += (val == zero) ? 0 : w;

    val = at(x0 + 1, y0 + 1).ToVectorf();
    w = a * b;
    sum_val += w * val;
    sum_w += (val == zero) ? 0 : w;

    val = at(x0 + 1, y0).ToVectorf();
    w = a * (1 - b);
    sum_val += w * val;
    sum_w += (val == zero) ? 0 : w;

    return sum_w == 0 ? VecType(0) : VecType::FromVectorf(sum_val / sum_w);
}

template<typename VecType>
__device__
    VecType
ImageCudaDevice<VecType>::BoxFilter2x2(int x, int y) {
#ifdef CUDA_DEBUG_ENABLE_ASSERTION
    assert(x >= 0 && x < width_);
    assert(y >= 0 && y < height_);
#endif

    int xp1 = min(width_ - 1, x + 1);
    int yp1 = min(height_ - 1, y + 1);

    auto sum_val = VecType::VecTypef::Zeros();
    sum_val += at(x, y).ToVectorf();
    sum_val += at(x, yp1).ToVectorf();
    sum_val += at(xp1, y).ToVectorf();
    sum_val += at(xp1, yp1).ToVectorf();

    sum_val *= 0.25f;
    return VecType::FromVectorf(sum_val);
}

template<typename VecType>
__device__
    VecType
ImageCudaDevice<VecType>::BoxFilter2x2WithHoles(
    int x, int y, float threshold) {
#ifdef CUDA_DEBUG_ENABLE_ASSERTION
    assert(x >= 0 && x < width_);
    assert(y >= 0 && y < height_);
#endif

    int xp1 = min(width_ - 1, x + 1);
    int yp1 = min(height_ - 1, y + 1);

    auto sum_val = VecType::VecTypef::Zeros();
    float cnt = 0.0;
    bool is_valid;
    VecType
    val0, val;
    VecType
    zero = VecType(0);

    val0 = at(x, y);
    if (val0 == zero) return zero;
    sum_val += val0.ToVectorf();
    cnt += 1.0f;

    /** Check neighbors **/
    val = at(x, yp1);
    is_valid = (val != zero && (val - val0).norm() < threshold);
    sum_val += is_valid ? val.ToVectorf() : VecType::VecTypef::Zeros();
    cnt += is_valid ? 1.0f : 0.0f;

    val = at(xp1, y);
    is_valid = (val != zero && (val - val0).norm() < threshold);
    sum_val += is_valid ? val.ToVectorf() : VecType::VecTypef::Zeros();
    cnt += is_valid ? 1.0f : 0.0f;

    val = at(xp1, yp1);
    is_valid = (val != zero && (val - val0).norm() < threshold);
    sum_val += is_valid ? val.ToVectorf() : VecType::VecTypef::Zeros();
    cnt += is_valid ? 1.0f : 0.0f;

    return VecType::FromVectorf(sum_val / cnt);
}

template<typename VecType>
__device__
    VecType
ImageCudaDevice<VecType>::GaussianFilter(int x, int y, int kernel_idx) {
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

    auto sum_val = VecType::VecTypef::Zeros();
    float sum_weight = 0;

    for (int xx = x_min; xx <= x_max; ++xx) {
        for (int yy = y_min; yy <= y_max; ++yy) {
            auto val = at(xx, yy).ToVectorf();
            float weight = kernel[abs(xx - x)] * kernel[abs(yy - y)];
            sum_val += val * weight;
            sum_weight += weight;
        }
    }

    sum_val /= sum_weight;

    return VecType::FromVectorf(sum_val);
}

template<typename VecType>
__device__
    VecType
ImageCudaDevice<VecType>::GaussianFilterWithHoles(
    int x, int y, int kernel_idx, float threshold) {
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

    /** If it is already a hole, leave it alone **/
    VecType
    zero = VecType(0);
    auto &val0 = at(x, y);
    if (val0 == zero) return zero;

    const int kernel_size = kernel_sizes[kernel_idx];
    const int kernel_size_2 = kernel_size >> 1;
    const float *kernel = gaussian_weights[kernel_idx];

    int x_min = max(0, x - kernel_size_2);
    int y_min = max(0, y - kernel_size_2);
    int x_max = min(width_ - 1, x + kernel_size_2);
    int y_max = min(height_ - 1, y + kernel_size_2);

    auto sum_val = VecType::VecTypef::Zeros();
    float sum_weight = 0;

    for (int xx = x_min; xx <= x_max; ++xx) {
        for (int yy = y_min; yy <= y_max; ++yy) {
            VecType
            val = at(xx, yy);
            auto valf = val.ToVectorf();

            /** TODO: Check it carefully **/
            float weight = (val == zero) //|| (val - val0).norm() > threshold
                           ? 0 : kernel[abs(xx - x)] * kernel[abs(yy - y)];
            sum_val += valf * weight;
            sum_weight += weight;
        }
    }

    /** Center is not zero, so sum_weight > 0 **/
    sum_val /= sum_weight;
    return VecType::FromVectorf(sum_val);
}

template<typename VecType>
__device__
    VecType
ImageCudaDevice<VecType>::BilateralFilter(
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

    auto center_val = at(x, y).ToVectorf();
    auto sum_val = VecType::VecTypef::Zeros();
    float sum_weight = 0;

    for (int xx = x_min; xx <= x_max; ++xx) {
        for (int yy = y_min; yy <= y_max; ++yy) {
            auto val = at(xx, yy).ToVectorf();
            float weight = kernel[abs(xx - x)] * kernel[abs(yy - y)];
            float value_diff = (val - center_val).norm() / val_sigma;
            weight *= expf(-value_diff);

            sum_val += val * weight;
            sum_weight += weight;
        }
    }

    sum_val /= sum_weight;
    return VecType::FromVectorf(sum_val);
}

template<typename VecType>
__device__
    VecType
ImageCudaDevice<VecType>::BilateralFilterWithHoles(
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

    VecType
    zero = VecType(0);
    if (at(x, y) == zero) return zero;

    const int kernel_size = kernel_sizes[kernel_idx];
    const int kernel_size_2 = kernel_size >> 1;
    const float *kernel = gaussian_weights[kernel_idx];

    int x_min = max(0, x - kernel_size_2);
    int y_min = max(0, y - kernel_size_2);
    int x_max = min(width_ - 1, x + kernel_size_2);
    int y_max = min(height_ - 1, y + kernel_size_2);

    auto center_valf = at(x, y).ToVectorf();
    auto sum_val = VecType::VecTypef::Zeros();
    float sum_weight = 0;

    for (int xx = x_min; xx <= x_max; ++xx) {
        for (int yy = y_min; yy <= y_max; ++yy) {
            auto val = at(xx, yy);
            auto valf = val.ToVectorf();
            float weight = kernel[abs(xx - x)] * kernel[abs(yy - y)];
            float value_diff = (valf - center_valf).norm() / val_sigma;
            weight *= expf(-value_diff * value_diff);

            sum_val += valf * weight;
            sum_weight += (val == zero) ? 0 : weight;
        }
    }

    sum_val /= sum_weight;
    return VecType::FromVectorf(sum_val);
}

template<typename VecType>
__device__
    ImageCudaDevice<VecType>::Grad
ImageCudaDevice<VecType>::Sobel(int x, int y) {
#ifdef CUDA_DEBUG_ENABLE_ASSERTION
    assert(x >= 1 && x < width_ - 1);
    assert(y >= 1 && y < height_ - 1);
#endif

    auto Iumvm = at(x - 1, y - 1).ToVectorf();
    auto Iumv0 = at(x - 1, y).ToVectorf();
    auto Iumvp = at(x - 1, y + 1).ToVectorf();
    auto Iu0vm = at(x, y - 1).ToVectorf();
    auto Iu0vp = at(x, y + 1).ToVectorf();
    auto Iupvm = at(x + 1, y - 1).ToVectorf();
    auto Iupv0 = at(x + 1, y).ToVectorf();
    auto Iupvp = at(x + 1, y + 1).ToVectorf();

    return {
        (Iupvm - Iumvm) + (Iupv0 - Iumv0) * 2 + (Iupvp - Iumvp),
        (Iumvp - Iumvm) + (Iu0vp - Iu0vm) * 2 + (Iupvp - Iupvm)
    };
}

/**
 * It is a little bit strict ...
 * It will filter out all the (geometric) edges.
 * If it does not work, fall back to Sobel
 */
template<typename VecType>
__device__
    ImageCudaDevice<VecType>::Grad
ImageCudaDevice<VecType>::SobelWithHoles(int x, int y) {
#ifdef CUDA_DEBUG_ENABLE_ASSERTION
    assert(x >= 1 && x < width_ - 1);
    assert(y >= 1 && y < height_ - 1);
#endif

    auto zero = VecType::VecTypef(0);
    auto Iumvm = at(x - 1, y - 1).ToVectorf();
    auto Iumv0 = at(x - 1, y).ToVectorf();
    auto Iumvp = at(x - 1, y + 1).ToVectorf();
    auto Iu0vm = at(x, y - 1).ToVectorf();
    auto Iu0vp = at(x, y + 1).ToVectorf();
    auto Iupvm = at(x + 1, y - 1).ToVectorf();
    auto Iupv0 = at(x + 1, y).ToVectorf();
    auto Iupvp = at(x + 1, y + 1).ToVectorf();

    bool mask_corner = (Iumvm != zero) && (Iumvp != zero)
        && (Iupvm != zero) && (Iupvp != zero);
    bool mask_dx = mask_corner && (Iupv0 != zero) && (Iumv0 != zero);
    bool mask_dy = mask_corner && (Iu0vp != zero) && (Iu0vm != zero);
    return {
        mask_dx ?
        (Iupvm - Iumvm) + (Iupv0 - Iumv0) * 2 + (Iupvp - Iumvp) : zero,
        mask_dy ?
        (Iumvp - Iumvm) + (Iu0vp - Iu0vm) * 2 + (Iupvp - Iupvm) : zero
    };
}
} // cuda
} // open3d