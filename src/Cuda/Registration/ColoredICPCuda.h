//
// Created by wei on 1/11/19.
//

#pragma once

#include <Cuda/Common/JacobianCuda.h>
#include <Cuda/Geometry/PointCloudCuda.h>
#include <Cuda/Registration/TransformEstimationCuda.h>
#include <Open3D/Geometry/KDTreeFlann.h>

namespace open3d {
namespace cuda {

/* We don't want inheritance for cuda classes */
class TransformEstimationForColoredICPCudaDevice {
public:
    PointCloudCudaDevice source_;
    PointCloudCudaDevice target_;
    ArrayCudaDevice<Vector3f> target_color_gradient_;

    ArrayCudaDevice<float> results_;
    CorrespondenceSetCudaDevice correspondences_;

    TransformCuda transform_source_to_target_;

public:
    /* We need some coefficients here */
    float lambda_geometric_;
    float sqrt_coeff_I_;
    float sqrt_coeff_G_;

public:
    __DEVICE__ void ComputePointwiseJacobianAndResidual(
        int source_idx, int target_idx,
        Vector6f& jacobian_I, Vector6f &jacobian_G,
        float &residual_I, float &residual_G);

    __DEVICE__ void ComputePointwiseGradient(
        int idx, CorrespondenceSetCudaDevice &corres_for_color_gradient);
};

class TransformEstimationForColoredICPCuda : public TransformEstimationCuda {
public:
    std::shared_ptr<TransformEstimationForColoredICPCudaDevice>
        device_ = nullptr;

public:
    registration::TransformationEstimationType GetTransformationEstimationType()
    const override { return type_; };

    TransformEstimationForColoredICPCuda(float lambda_geometric = 0.968f);
    ~TransformEstimationForColoredICPCuda() override;

    void Create() override;
    void Release() override;
    void UpdateDevice() override;

    /** TODO: copy constructors **/

public:
    void Initialize(
        geometry::PointCloud &source,
        geometry::PointCloud &target,
        float max_correspondence_distance) override;
    RegistrationResultCuda ComputeResultsAndTransformation() override;

public:
    /** Computes color gradients
     * 1. Get correspondence matrix on CPU
     * 2. Compress the correspondence matrix
     * 3. Use the compressed correspondence matrix to build linear systems
     * and compute color gradients.
     * **/
    void ComputeColorGradients(
        geometry::PointCloud &target,
        geometry::KDTreeFlann &kdtree,
        const geometry::KDTreeSearchParamHybrid &search_param);

public:
    float lambda_geometric_;
    ArrayCuda<Vector3f> target_color_gradient_;

private:
    const registration::TransformationEstimationType type_ =
        registration::TransformationEstimationType::ColoredICP;
};

class TransformEstimationCudaForColoredICPKernelCaller {
public:
    static void ComputeColorGradeint(
        TransformEstimationForColoredICPCuda &estimation,
        CorrespondenceSetCuda &corres_for_color_gradient);

    static void ComputeResultsAndTransformation(
        TransformEstimationForColoredICPCuda &estimation);
};

__GLOBAL__
void ComputeColorGradientKernel(
    TransformEstimationForColoredICPCudaDevice estimation,
    CorrespondenceSetCudaDevice corres_for_color_gradient);

__GLOBAL__
void ComputeResultsAndTransformationKernel(
    TransformEstimationForColoredICPCudaDevice estimation);

} // cuda
} // open3d