//
// Created by wei on 1/21/19.
//

#pragma once

#include "FastGlobalRegistrationCuda.h"

namespace open3d {
namespace cuda {

__device__
void FastGlobalRegistrationCudaDevice::
ComputePointwiseJacobianAndResidual(
    int source_idx, int target_idx,
    Vector6f &jacobian_x, Vector6f &jacobian_y, Vector6f &jacobian_z,
    Vector3f &residual, float &lij) {

    Vector3f p = target_.points()[target_idx];
    Vector3f q = source_.points()[source_idx];
    Vector3f rpq = p - q;

    lij = par_ / (rpq.dot(rpq) + par_);

    residual = lij * rpq;

    jacobian_x(0) = jacobian_x(4) = jacobian_x(5) = 0;
    jacobian_x(1) = -q(2) * lij;
    jacobian_x(2) = q(1) * lij;
    jacobian_x(3) = -lij;

    jacobian_y(1) = jacobian_y(3) = jacobian_y(5) = 0;
    jacobian_y(2) = -q(0) * lij;
    jacobian_y(0) = q(2) * lij;
    jacobian_y(4) = -lij;

    jacobian_z(2) = jacobian_z(3) = jacobian_z(4) = 0;
    jacobian_z(0) = -q(1) * lij;
    jacobian_z(1) = q(0) * lij;
    jacobian_z(5) = -lij;
}

} // cuda
} // open3d
