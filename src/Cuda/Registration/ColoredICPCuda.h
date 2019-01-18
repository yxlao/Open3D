//
// Created by wei on 1/11/19.
//

#pragma once

#include <Cuda/Common/JacobianCuda.h>
#include <Cuda/Geometry/PointCloudCuda.h>
#include <Cuda/Registration/TransformEstimationCuda.h>
#include <Core/Geometry/KDTreeFlann.h>

namespace open3d {
namespace cuda {

/* We don't want inheritance for cuda classes */
class TransformEstimationCudaForColoredICPDevice {
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
};

class TransformEstimationCudaForColoredICP : public TransformEstimationCuda {
public:
    std::shared_ptr<TransformEstimationCudaForColoredICPDevice>
        server_ = nullptr;

public:
    TransformationEstimationType GetTransformationEstimationType()
    const override { return type_; };

    TransformEstimationCudaForColoredICP(float lambda_geometric = 0.968f);
    ~TransformEstimationCudaForColoredICP() override;

    void Create();
    void Release();
    void UpdateServer() override;

    /** TODO: copy constructors **/

public:
    void Initialize(PointCloud &source, PointCloud &target,
                    float max_correspondence_distance) override;
    RegistrationResultCuda ComputeResultsAndTransformation() override;

public:
    /** Computes color gradients
     * 1. Get correspondence matrix on CPU
     * 2. Compress the correspondence matrix
     * 3. Use the compressed correspondence matrix to build linear systems
     * and compute color gradients.
     * **/
    void ComputeColorGradients(PointCloud &target,
                               KDTreeFlann &kdtree,
                               const KDTreeSearchParamHybrid &search_param);

public:
    float lambda_geometric_;
    ArrayCuda<Vector3f> target_color_gradient_;

private:
    const TransformationEstimationType type_ =
        TransformationEstimationType::ColoredICP;
};

class TransformEstimationCudaForColoredICPKernelCaller {
public:
    static void ComputeColorGradeintKernelCaller(
        TransformEstimationCudaForColoredICP &estimation,
        CorrespondenceSetCuda &corres_for_color_gradient);

    static void ComputeResultsAndTransformationKernelCaller(
        TransformEstimationCudaForColoredICP &estimation);
};

__GLOBAL__
void ComputeColorGradientKernel(
    TransformEstimationCudaForColoredICPDevice estimation,
    CorrespondenceSetCudaDevice corres_for_color_gradient);

__GLOBAL__
void ComputeResultsAndTransformationKernel(
    TransformEstimationCudaForColoredICPDevice estimation);

} // cuda
} // open3d