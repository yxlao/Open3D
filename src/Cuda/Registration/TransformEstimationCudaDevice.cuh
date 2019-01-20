//
// Created by wei on 1/17/19.
//

#include "TransformEstimationCuda.h"

namespace open3d {
namespace cuda {
__device__
void TransformEstimationPointToPointCudaDevice
::ComputePointwiseJacobianAndResidual(
    int source_idx, int target_idx,
    Vector6f &jacobian, float &residual) {
    /* Re-implement umeyama
     * http://edge.cs.drexel.edu/Dmitriy/Matching_and_Metrics/Umeyama/um.pdf */

}

__device__
void TransformEstimationPointToPlaneCudaDevice
::ComputePointwiseJacobianAndResidual(
    int source_idx, int target_idx,
    Vector6f &jacobian, float &residual) {

    const Vector3f &vs = source_.points()[source_idx];

    const Vector3f &vt = target_.points()[target_idx];
    const Vector3f &nt = target_.normals()[target_idx];

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