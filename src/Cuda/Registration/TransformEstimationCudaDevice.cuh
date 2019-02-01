//
// Created by wei on 1/17/19.
//

#include "TransformEstimationCuda.h"
#include <Cuda/Common/JacobianCuda.h>

namespace open3d {
namespace cuda {
__device__
void TransformEstimationPointToPointCudaDevice
::ComputePointwiseStatistics(
    int source_idx, int target_idx,
    Matrix3f &Sigma, float &source_sigma2, float &residual) {
    const Vector3f &vs = source_.points_[source_idx];
    const Vector3f &vt = target_.points_[target_idx];

    const Vector3f ds(vs(0) - source_mean_(0),
                      vs(1) - source_mean_(1),
                      vs(2) - source_mean_(2));
    const Vector3f dt(vt(0) - target_mean_(0),
                      vt(1) - target_mean_(1),
                      vt(2) - target_mean_(2));
    const Vector3f dst(vs(0) - vt(0), vs(1) - vt(1), vs(2) - vt(2));

#pragma unroll 1
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
void TransformEstimationPointToPlaneCudaDevice
::ComputePointwiseJacobianAndResidual(
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

} // cuda
} // open3d