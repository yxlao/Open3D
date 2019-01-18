//
// Created by wei on 10/1/18.
//

#pragma once

#include "RGBDOdometryCuda.h"

#include <Cuda/Common/UtilsCuda.h>
#include <Cuda/Geometry/ImageCudaDevice.cuh>
#include <Cuda/Container/ArrayCudaDevice.cuh>

namespace open3d {
namespace cuda {
/**
 * Server end
 */
template<size_t N>
__device__
bool RGBDOdometryCudaDevice<N>::ComputePixelwiseCorrespondenceAndResidual(
    int x_source, int y_source, size_t level,
    int &x_target, int &y_target,
    Vector3f &X_target,
    float &residual_I, float &residual_D) {

    /** Check 1: depth valid in source? **/
    float d_source = source_[level].depth().at(x_source, y_source)(0);
    bool mask = IsValidDepth(d_source);
    if (!mask) return false;

    /** Check 2: reprojected point in image? **/
    X_target = transform_source_to_target_
        * intrinsics_[level].InverseProjectPixel(
            Vector2i(x_source, y_source), d_source);

    Vector2f p_warpedf = intrinsics_[level].ProjectPoint(X_target);
    mask = intrinsics_[level].IsPixelValid(p_warpedf);
    if (!mask) return false;

    Vector2i p_warped(int(p_warpedf(0) + 0.5f), int(p_warpedf(1) + 0.5f));

    /** Check 3: depth valid in target? Occlusion? -> 1ms **/
    float d_target = target_[level].depth().at(p_warped(0), p_warped(1))(0);
    mask = IsValidDepth(d_target) && IsValidDepthDiff(d_target - X_target(2));
    if (!mask) return false;

    x_target = p_warped(0);
    y_target = p_warped(1);
    residual_I = sqrt_coeff_I_ * (
        target_[level].intensity().at(x_target, y_target)(0) -
            source_[level].intensity().at(x_source, y_source)(0));
    residual_D = sqrt_coeff_D_ * (d_target - X_target(2));

    return true;
}

template<size_t N>
__device__
bool RGBDOdometryCudaDevice<N>::ComputePixelwiseJacobian(
    int x_target, int y_target, size_t level,
    const Vector3f &X_target,
    Vector6f &jacobian_I, Vector6f &jacobian_D) {

    /********** Phase 2: Build linear system **********/
    /** Checks passed, let's rock! -> 3ms, can be 2ms faster if we don't use
     * interpolation
     *  \partial D(p_warped) \partial p_warped: [dx_D, dy_D] at p_warped, 1x2
     *  \partial I(p_warped) \partial p_warped: [dx_I, dy_I] at p_warped, 1x2
     *  \partial X.z \partial X: [0, 0, 1], 1x3
     *  \partial p_warped \partial X: [fx/Z, 0, -fx X/Z^2;
     *                                 0, fy/Z, -fy Y/Z^2]            2x3
     *  \partial X \partial \xi: [I | -[X]^] = [1 0 0 0  Z -Y;
     *                                          0 1 0 -Z 0 X;
     *                                          0 0 1 Y -X 0]         3x6
     * J_I = (d I(p_warped) / d p_warped) (d p_warped / d X) (d X / d \xi)
     * J_D = (d D(p_warped) / d p_warped) (d p_warped / d X) (d X / d \xi)
     *     - (d X.z / d X) (d X / d \xi)
     */
    const float kSobelFactor = 0.125f;
    float dx_I = kSobelFactor * target_dx_[level].intensity().at(
        x_target, y_target)(0);
    float dy_I = kSobelFactor * target_dy_[level].intensity().at(
        x_target, y_target)(0);

    float dx_D = kSobelFactor * target_dx_[level].depth().at(
        x_target, y_target)(0);
    float dy_D = kSobelFactor * target_dy_[level].depth().at(
        x_target, y_target)(0);

    float fx = intrinsics_[level].fx_;
    float fy = intrinsics_[level].fy_;
    float inv_Z = 1.0f / X_target(2);
    float fx_on_Z = fx * inv_Z;
    float fy_on_Z = fy * inv_Z;

    float c0 = dx_I * fx_on_Z;
    float c1 = dy_I * fy_on_Z;
    float c2 = -(c0 * X_target(0) + c1 * X_target(1)) * inv_Z;

    jacobian_I(0) = sqrt_coeff_I_ * (-X_target(2) * c1 + X_target(1) * c2);
    jacobian_I(1) = sqrt_coeff_I_ * (X_target(2) * c0 - X_target(0) * c2);
    jacobian_I(2) = sqrt_coeff_I_ * (-X_target(1) * c0 + X_target(0) * c1);
    jacobian_I(3) = sqrt_coeff_I_ * c0;
    jacobian_I(4) = sqrt_coeff_I_ * c1;
    jacobian_I(5) = sqrt_coeff_I_ * c2;

    float d0 = dx_D * fx_on_Z;
    float d1 = dy_D * fy_on_Z;
    float d2 = -(d0 * X_target(0) + d1 * X_target(1)) * inv_Z;

    jacobian_D(0) = sqrt_coeff_D_ *
        ((-X_target(2) * d1 + X_target(1) * d2) - X_target(1));
    jacobian_D(1) = sqrt_coeff_D_ *
        ((X_target(2) * d0 - X_target(0) * d2) + X_target(0));
    jacobian_D(2) = sqrt_coeff_D_ *
        (-X_target(1) * d0 + X_target(0) * d1);
    jacobian_D(3) = sqrt_coeff_D_ * d0;
    jacobian_D(4) = sqrt_coeff_D_ * d1;
    jacobian_D(5) = sqrt_coeff_D_ * (d2 - 1.0f);
    return true;
}

template<size_t N>
__device__
void RGBDOdometryCudaDevice<N>::ComputePixelwiseJtJAndJtr(
    const Vector6f &jacobian_I, const Vector6f &jacobian_D,
    const float &residual_I, const float &residual_D,
    HessianCuda<6> &JtJ, Vector6f &Jtr) {

    int cnt = 0;
#pragma unroll 1
    for (int i = 0; i < 6; ++i) {
#pragma unroll 1
        for (int j = i; j < 6; ++j) {
            JtJ(cnt++) = jacobian_I(i) * jacobian_I(j)
                + jacobian_D(i) * jacobian_D(j);
        }
        Jtr(i) = jacobian_I(i) * residual_I + jacobian_D(i) * residual_D;
    }
}
} // cuda
} // open3d