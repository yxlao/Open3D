//
// Created by wei on 4/4/19.
//

#include "ScalableVolumeRegistrationCuda.h"
#include <Cuda/Integration/UniformTSDFVolumeCuda.h>

namespace open3d {
namespace cuda {
__device__
bool ScalableVolumeRegistrationCudaDevice::ComputeVoxelwiseJacobianAndResidual(
    const Vector3i &Xlocal_target, const Vector3i &Xsv_target,
    UniformTSDFVolumeCudaDevice *subvolume_target, int subvolume_target_idx,
    Vector6f &jacobian, float &residual) {

    uchar w_target = subvolume_target->weight(Xlocal_target);
    if (w_target == 0) return false;

    float d_target = subvolume_target->tsdf(Xlocal_target);
    if (fabsf(d_target) >= 0.5f) return false;

    Vector3i Xglobal_target = target_.voxel_local_to_global(
        Xlocal_target, Xsv_target);
    Vector3f Xtarget = target_.voxelf_to_world(Xglobal_target.cast<float>());

    Vector3f Xglobalf_source =
        source_.world_to_voxelf(trans_target_to_source_ * Xtarget);
    Vector3i Xglobal_source = Xglobalf_source.cast<int>();


    uchar w_source = source_.weight(Xglobal_source);
    if (w_source == 0) return false;

    float d_source = source_.tsdf(Xglobal_source);
    if (fabsf(d_target) >= 0.5f) return false;

    float sqrt_w = 1; //sqrtf(w_target / 255.0f);
    residual = sqrt_w * (d_target - d_source);

    Vector3f grad_target = target_property_.gradient(
        Xsv_target, subvolume_target_idx);

    float c = sqrt_w / target_.voxel_length_;
    jacobian(0) = c * (-Xtarget(2) * grad_target(1) + Xtarget(1) * grad_target(2));
    jacobian(1) = c * (Xtarget(2) * grad_target(0) - Xtarget(0) * grad_target(2));
    jacobian(2) = c * (-Xtarget(1) * grad_target(0) + Xtarget(0) * grad_target(1));
    jacobian(3) = c * grad_target(0);
    jacobian(4) = c * grad_target(1);
    jacobian(5) = c * grad_target(2);

    return true;
}
}
}