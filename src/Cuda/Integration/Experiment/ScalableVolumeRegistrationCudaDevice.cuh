//
// Created by wei on 4/4/19.
//

#include "ScalableVolumeRegistrationCuda.h"
#include <Cuda/Integration/UniformTSDFVolumeCuda.h>

namespace open3d {
namespace cuda {
__device__
bool ScalableVolumeRegistrationCudaDevice::ComputeVoxelwiseJacobianAndResidual(
    const Vector3i &Xlocal_source, const Vector3i &Xsv_source,
    UniformTSDFVolumeCudaDevice *subvolume_source, int subvolume_source_idx,
    Vector6f &jacobian, float &residual) {

    uchar w_source = subvolume_source->weight(Xlocal_source);
    if (w_source == 0) return false;

    float d_source = subvolume_source->tsdf(Xlocal_source);
    if (fabsf(d_source) >= 0.1f) return false;

    Vector3i Xglobal_source = source_.voxel_local_to_global(
        Xlocal_source, Xsv_source);
    Vector3f Xsource = source_.voxelf_to_world(Xglobal_source.cast<float>());

    Vector3f Xglobalf_target =
        target_.world_to_voxelf(trans_source_to_target_ * Xsource);
    Vector3i Xglobal_target = Xglobalf_target.cast<int>();

    uchar w_target = target_.weight(Xglobal_target);
    if (w_target == 0) return false;

    float d_target = target_.tsdf(Xglobal_target);
//    if (fabsf(d_target) >= 0.1f) return false;

    float sqrt_w = 1; //sqrtf(w_target / 255.0f);
    residual = sqrt_w * (d_source - d_target);

    Vector3f grad_source = source_property_.gradient(
        Xsv_source, subvolume_source_idx);

//    grad_source = 0.1f * Vector3f::Ones();
    float c = sqrt_w / source_.voxel_length_;
    jacobian(0) = c * (-Xsource(2) * grad_source(1) + Xsource(1) * grad_source(2));
    jacobian(1) = c * (Xsource(2) * grad_source(0) - Xsource(0) * grad_source(2));
    jacobian(2) = c * (-Xsource(1) * grad_source(0) + Xsource(0) * grad_source(1));
    jacobian(3) = c * grad_source(0);
    jacobian(4) = c * grad_source(1);
    jacobian(5) = c * grad_source(2);

    return true;
}
}
}