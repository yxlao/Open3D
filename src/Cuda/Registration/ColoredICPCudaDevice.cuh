//
// Created by wei on 1/15/19.
//

#pragma once

#include "ColoredICPCuda.h"

namespace open3d {
namespace cuda {

__device__
void TransformEstimationCudaForColoredICPDevice::
ComputePointwiseJacobianAndResidual(
    int source_idx, int target_idx,
    Vector6f &jacobian_I, Vector6f &jacobian_G,
    float &residual_I, float &residual_G) {

    const Vector3f &vs = source_.points_[source_idx];
    const Vector3f &cs = source_.colors_[source_idx];
    float is = (cs(0) + cs(1) + cs(2)) / 3.0f;

    const Vector3f &vt = target_.points_[target_idx];
    const Vector3f &ct = target_.colors_[target_idx];
    float it = (ct(0) + ct(1) + ct(2)) / 3.0f;
    const Vector3f &dit = target_color_gradient_[target_idx];
    const Vector3f &nt = target_.normals_[target_idx];

    Vector3f vs_proj = vs - (vs - vt).dot(nt) * nt;
    float is0_proj = dit.dot(vs_proj - vt) + it;

    jacobian_G(0) = sqrt_coeff_G_ * (-vs(2) * nt(1) + vs(1) * nt(2));
    jacobian_G(1) = sqrt_coeff_G_ * (vs(2) * nt(0) - vs(0) * nt(2));
    jacobian_G(2) = sqrt_coeff_G_ * (-vs(1) * nt(0) + vs(0) * nt(1));
    jacobian_G(3) = sqrt_coeff_G_ * nt(0);
    jacobian_G(4) = sqrt_coeff_G_ * nt(1);
    jacobian_G(5) = sqrt_coeff_G_ * nt(2);
    residual_G    = sqrt_coeff_G_ * (vs - vt).dot(nt);

    Vector3f ditM = dit.dot(nt) * nt - dit;
    jacobian_I(0) = sqrt_coeff_I_ * (-vs(2) * ditM(1) + vs(1) * ditM(2));
    jacobian_I(1) = sqrt_coeff_I_ * (vs(2) * ditM(0) - vs(0) * ditM(2));
    jacobian_I(2) = sqrt_coeff_I_ * (-vs(1) * ditM(0) + vs(0) * ditM(1));
    jacobian_I(3) = sqrt_coeff_I_ * ditM(0);
    jacobian_I(4) = sqrt_coeff_I_ * ditM(1);
    jacobian_I(5) = sqrt_coeff_I_ * ditM(2);
    residual_I    = sqrt_coeff_I_ * (is - is0_proj);
}
} // cuda
} // open3d