//
// Created by wei on 1/10/19.
//

#pragma once

#include <Open3D/Registration/Registration.h>
#include <Open3D/Geometry/PointCloud.h>
#include <Open3D/Geometry/KDTreeFlann.h>

#include <Cuda/Container/ArrayCuda.h>
#include <Cuda/Geometry/PointCloudCuda.h>
#include <CorrespondenceSetCuda.h>

namespace open3d {

namespace cuda {

class RegistrationResultCudaR {
public:
    RegistrationResultCudaR(const Eigen::Matrix4d &transformation =
    Eigen::Matrix4d::Identity()) : transformation_(transformation),
                                   inlier_rmse_(0.0), fitness_(0.0) {}
    ~RegistrationResultCudaR() {}

public:
    Eigen::Matrix4d transformation_;
    CorrespondenceSetCuda correspondences_;
    float inlier_rmse_;
    float fitness_;
};

class RegistrationCudaDeviceR {
public:
    PointCloudCudaDevice source_;
    PointCloudCudaDevice target_;

    ArrayCudaDevice<float> results_;
    CorrespondenceSetCudaDevice correspondences_;

    TransformCuda transform_source_to_target_;

public: /* reserved for colored ICP */
    ArrayCudaDevice<Vector3f> target_color_gradient_;
    float sqrt_coeff_I_;
    float sqrt_coeff_G_;

public:
    /** Colored ICP **/
    __DEVICE__ void ComputePointwiseColoredJacobianAndResidual(
        int source_idx, int target_idx,
        Vector6f &jacobian_I, Vector6f &jacobian_G,
        float &residual_I, float &residual_G);

    /** PointToPlane **/
    __DEVICE__ void ComputePointwisePointToPlaneJacobianAndResidual(
        int source_idx, int target_idx,
        Vector6f &jacobian, float &residual);

    /** PointToPoint **/
    __DEVICE__ void ComputePointwisePointToPointSigmaAndResidual(
        int source_idx, int target_idx,
        const Vector3f &mean_source, const Vector3f &mean_target,
        Matrix3f &Sigma, float &source_sigma2, float &residual);

    /** Shared **/
    __DEVICE__ void ComputePixelwiseInformationJacobian(
        const Vector3f &point,
        Vector6f &jacobian_x, Vector6f &jacobian_y, Vector6f &jacobian_z);

    __DEVICE__ void ComputePointwiseColorGradient(
        int idx, CorrespondenceSetCudaDevice &corres_for_color_gradient);

};

class RegistrationCudaR {
public:
    std::shared_ptr<RegistrationCudaDeviceR> device_;
    Eigen::Matrix4d transform_source_to_target_;

    registration::TransformationEstimationType type_;

    /** For GPU **/
    PointCloudCuda source_;
    PointCloudCuda target_;

    /** 1-nn, source x 1 (in target) **/
    CorrespondenceSetCuda correspondences_;
    float max_correspondence_distance_;

    /** For CPU NN search **/
    geometry::PointCloud source_cpu_;
    geometry::PointCloud target_cpu_;
    geometry::KDTreeFlann kdtree_;
    Eigen::MatrixXi corres_matrix_;

    /** Build linear system **/
    ArrayCuda<float> results_;

    /* For colored ICP */
    float lambda_geometric_;
    ArrayCuda<Vector3f> target_color_gradient_;

public:
    /* Life cycle */
    explicit RegistrationCudaR(
        const registration::TransformationEstimationType &type);
    ~RegistrationCudaR();

    void Create(const registration::TransformationEstimationType &type);
    void Release();
    void UpdateDevice();

public:
    /* Preparation */
    void Initialize(geometry::PointCloud &source,
                    geometry::PointCloud &target,
                    float max_correspondence_distance,
                    const Eigen::Matrix4d &init = Eigen::Matrix4d::Identity());

    /* High level API */
    RegistrationResultCudaR ComputeICP(int iter = 60);
    Eigen::Matrix6d ComputeInformationMatrix();

    /* Components for ICP */
    RegistrationResultCudaR DoSingleIteration(int iter);
    void GetCorrespondences();
    void TransformSourcePointCloud(const Eigen::Matrix4d &source_to_target);
    void ExtractResults(Eigen::Matrix6d &JtJ,
                        Eigen::Vector6d &Jtr,
                        float &rmse);
    RegistrationResultCudaR DoSingleIteration();

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

    /** **/
    RegistrationResultCudaR Umeyama();

};

class RegistrationCudaKernelCallerR {
public:
    static void ComputeColorGradeint(
        RegistrationCudaR &registration,
        CorrespondenceSetCuda &corres_for_color_gradient);

    static void BuildLinearSystemForColoredICP(
        RegistrationCudaR &registration);

    static void BuildLinearSystemForPointToPlaneICP(
        RegistrationCudaR &registration);

    static void ComputeSumForPointToPointICP(
        RegistrationCudaR &registration);
    static void BuildLinearSystemForPointToPointICP(
        RegistrationCudaR &registration,
        const Vector3f &mean_source, const Vector3f &mean_target);

    static void ComputeInformationMatrix(RegistrationCudaR &estimation);
};

__GLOBAL__ void ComputeColorGradientKernel(
    RegistrationCudaDeviceR registration,
    CorrespondenceSetCudaDevice corres_for_color_gradient);

__GLOBAL__ void BuildLinearSystemForColoredICPKernel(
    RegistrationCudaDeviceR registration);

__GLOBAL__ void BuildLinearSystemForPointToPlaneICPKernel(
    RegistrationCudaDeviceR registration);

__GLOBAL__ void ComputeSumForPointToPointICPKernel(
    RegistrationCudaDeviceR registration);
__GLOBAL__ void BuildLinearSystemForPointToPointICPKernel(
    RegistrationCudaDeviceR registration,
    const Vector3f mean_source, const Vector3f mean_target);

__GLOBAL__ void ComputeInformationMatrixKernel(
    RegistrationCudaDeviceR registration);

} // cuda
} // open3d