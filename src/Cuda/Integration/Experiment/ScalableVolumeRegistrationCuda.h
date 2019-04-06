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
    ScalableTSDFVolumeProcessorCudaDevice source_property_;

    ArrayCudaDevice<float> results_;

    TransformCuda trans_source_to_target_;

    __DEVICE__ bool ComputeVoxelwiseJacobianAndResidual(
        const Vector3i &Xlocal_source, const Vector3i &Xsv_source,
        UniformTSDFVolumeCudaDevice* subvolume_source, int subvolume_source_idx,
        Vector6f &jacobian, float &residual);
};

class ScalableVolumeRegistrationCuda {
public:
    int source_active_subvolumes_;

    std::shared_ptr<ScalableVolumeRegistrationCudaDevice> device_ = nullptr;

    ScalableVolumeRegistrationCuda();
    ~ScalableVolumeRegistrationCuda();

    void Create();
    void Release();

    void UpdateDevice();

public:
    ScalableTSDFVolumeCuda source_;
    ScalableTSDFVolumeCuda target_;

    ScalableTSDFVolumeProcessorCuda source_property_;

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


