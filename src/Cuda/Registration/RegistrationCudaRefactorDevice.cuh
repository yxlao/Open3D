//
// Created by wei on 3/19/19.
//

#pragma once

#include "RegistrationCudaRefactor.h"

namespace open3d {
namespace cuda {

__device__
void RegistrationCudaDeviceR::ComputePointwiseColoredJacobianAndResidual(
    int source_idx,
    int target_idx,
    Vector6f &jacobian_I,
    Vector6f &jacobian_G,
    float &residual_I,
    float &residual_G) {

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
    residual_G = sqrt_coeff_G_ * (vs - vt).dot(nt);

    Vector3f ditM = dit.dot(nt) * nt - dit;
    jacobian_I(0) = sqrt_coeff_I_ * (-vs(2) * ditM(1) + vs(1) * ditM(2));
    jacobian_I(1) = sqrt_coeff_I_ * (vs(2) * ditM(0) - vs(0) * ditM(2));
    jacobian_I(2) = sqrt_coeff_I_ * (-vs(1) * ditM(0) + vs(0) * ditM(1));
    jacobian_I(3) = sqrt_coeff_I_ * ditM(0);
    jacobian_I(4) = sqrt_coeff_I_ * ditM(1);
    jacobian_I(5) = sqrt_coeff_I_ * ditM(2);
    residual_I = sqrt_coeff_I_ * (is - is0_proj);
}

__device__
void RegistrationCudaDeviceR::ComputePointwisePointToPlaneJacobianAndResidual(
    int source_idx, int target_idx,
    Vector6f &jacobian, float &residual) {

    const Vector3f &vs = source_.points_[source_idx];

    const Vector3f &vt = target_.points_[target_idx];
    const Vector3f &nt = target_.normals_[target_idx];

    jacobian(0) = (-vs(2) * nt(1) + vs(1) * nt(2));
    jacobian(1) = (vs(2) * nt(0) - vs(0) * nt(2));
    jacobian(2) = (-vs(1) * nt(0) + vs(0) * nt(1));
    jacobian(3) = nt(0);
    jacobian(4) = nt(1);
    jacobian(5) = nt(2);
    residual    = (vs - vt).dot(nt);
}

__device__
void RegistrationCudaDeviceR::ComputePointwisePointToPointSigmaAndResidual(
    int source_idx, int target_idx,
    const Vector3f &mean_source, const Vector3f &mean_target,
    Matrix3f &Sigma, float &source_sigma2, float &residual) {
    const Vector3f &vs = source_.points_[source_idx];
    const Vector3f &vt = target_.points_[target_idx];

    const Vector3f ds(vs(0) - mean_source(0),
                      vs(1) - mean_source(1),
                      vs(2) - mean_source(2));
    const Vector3f dt(vt(0) - mean_target(0),
                      vt(1) - mean_target(1),
                      vt(2) - mean_target(2));
    const Vector3f dst(vs(0) - vt(0), vs(1) - vt(1), vs(2) - vt(2));

    for (int i = 0; i < 3; ++i) {
#pragma unroll 1
        for (int j = 0; j < 3; ++j) {
            Sigma(i, j) = dt(i) * ds(j);
        }
    }

    source_sigma2 = ds.dot(ds);
    residual = dst.dot(dst);
}


__device__
void RegistrationCudaDeviceR::ComputePixelwiseInformationJacobian(
    const Vector3f &point,
    Vector6f &jacobian_x, Vector6f &jacobian_y, Vector6f &jacobian_z) {
    jacobian_x(0) = jacobian_x(4) = jacobian_x(5) = 0;
    jacobian_x(1) = point(2);
    jacobian_x(2) = -point(1);
    jacobian_x(3) = 1;

    jacobian_y(1) = jacobian_y(3) = jacobian_y(5) = 0;
    jacobian_y(0) = -point(2);
    jacobian_y(2) = point(0);
    jacobian_y(4) = 1.0f;

    jacobian_z(2) = jacobian_z(3) = jacobian_z(4) = 0;
    jacobian_z(0) = point(1);
    jacobian_z(1) = -point(0);
    jacobian_z(5) = 1.0f;
}


__device__
void RegistrationCudaDeviceR::ComputePointwiseColorGradient(
    int idx, CorrespondenceSetCudaDevice &corres_for_color_gradient) {
    int i = corres_for_color_gradient.indices_[idx];

    Vector3f &vt = target_.points_[i];
    Vector3f &nt = target_.normals_[i];
    Vector3f &color = target_.colors_[i];
    float it = (color(0) + color(1) + color(2)) / 3.0f;

    Matrix3f AtA(0);
    Vector3f Atb(0);

    int nn = 0, max_nn = corres_for_color_gradient.nn_count_[idx];
    for (int j = 1; j < max_nn; ++j) {
        int adj_idx = corres_for_color_gradient.matrix_(j, i);
        if (adj_idx == -1) break;

        Vector3f &vt_adj = target_.points_[adj_idx];
        Vector3f vt_proj = vt_adj - (vt_adj - vt).dot(nt) * nt;
        Vector3f &color_adj = target_.colors_[adj_idx];
        float it_adj = (color_adj(0) + color_adj(1) + color_adj(2)) / 3.0f;

        float a0 = vt_proj(0) - vt(0);
        float a1 = vt_proj(1) - vt(1);
        float a2 = vt_proj(2) - vt(2);
        float b = it_adj - it;

        AtA(0, 0) += a0 * a0;
        AtA(0, 1) += a0 * a1;
        AtA(0, 2) += a0 * a2;
        AtA(1, 1) += a1 * a1;
        AtA(1, 2) += a1 * a2;
        AtA(2, 2) += a2 * a2;
        Atb(0) += a0 * b;
        Atb(1) += a1 * b;
        Atb(2) += a2 * b;

        ++nn;
    }

    /* orthogonal constraint */
    float nn2 = nn * nn;
    AtA(0, 0) += nn2 * nt(0) * nt(0);
    AtA(0, 1) += nn2 * nt(0) * nt(1);
    AtA(0, 2) += nn2 * nt(0) * nt(2);
    AtA(1, 1) += nn2 * nt(1) * nt(1);
    AtA(1, 2) += nn2 * nt(1) * nt(2);
    AtA(2, 2) += nn2 * nt(2) * nt(2);

    /* Symmetry */
    AtA(1, 0) = AtA(0, 1);
    AtA(2, 0) = AtA(0, 2);
    AtA(2, 1) = AtA(1, 2);

    target_color_gradient_[i] = AtA.ldlt().Solve(Atb);
}
} // cuda
} // open3d