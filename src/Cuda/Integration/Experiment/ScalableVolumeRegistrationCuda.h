//
// Created by wei on 4/4/19.
//

#pragma once

#include <Cuda/Integration/ScalableTSDFVolumeCuda.h>
#include <Cuda/Integration/Experiment/ScalableTSDFVolumeProcessorCuda.h>
#include <Cuda/Registration/RegistrationCuda.h>

namespace open3d {
namespace cuda {

/** Note: we have to use IC,
 *  i.e., iterate over target active subvolumes.
 *  Otherwise we cannot locate gradient correctly. **/
class ScalableVolumeRegistrationCudaDevice {
public:
    ScalableTSDFVolumeCudaDevice source_;
    ScalableTSDFVolumeCudaDevice target_;
    ScalableTSDFVolumeProcessorCudaDevice target_property_;

    ArrayCudaDevice<float> results_;

    TransformCuda trans_source_to_target_;
    TransformCuda trans_target_to_source_;

    __DEVICE__ bool ComputeVoxelwiseJacobianAndResidual(
        const Vector3i &Xlocal_target, const Vector3i &Xsv_target,
        UniformTSDFVolumeCudaDevice* subvolume_target, int subvolume_target_idx,
        Vector6f &jacobian, float &residual);
};

class ScalableVolumeRegistrationCuda {
public:
    int target_active_subvolumes_;

    std::shared_ptr<ScalableVolumeRegistrationCudaDevice> device_ = nullptr;

    ScalableVolumeRegistrationCuda();
    ~ScalableVolumeRegistrationCuda();

    void Create();
    void Release();

    void UpdateDevice();

public:
    ScalableTSDFVolumeCuda source_;
    ScalableTSDFVolumeCuda target_;
    ScalableTSDFVolumeProcessorCuda target_property_;
    ArrayCuda<float> results_;

    Eigen::Matrix4d trans_source_to_target_;

public:
    void Initialize(ScalableTSDFVolumeCuda &source,
                    ScalableTSDFVolumeCuda &target,
                    const Eigen::Matrix4d &init = Eigen::Matrix4d::Identity());
    RegistrationResultCuda DoSingleIteration(int iter);
    RegistrationResultCuda BuildAndSolveLinearSystem();
    void ExtractResults(
        Eigen::Matrix6d &JtJ, Eigen::Vector6d &Jtr, float &rmse, float &inlier);
};

class ScalableVolumeRegistrationCudaKernelCaller {
public:
    static void BuildLinearSystem(ScalableVolumeRegistrationCuda &registration);
};

__GLOBAL__
void BuildLinearSystemKernel(ScalableVolumeRegistrationCudaDevice registration);

}
}


