//
// Created by wei on 10/1/18.
//

#pragma once

#include "OdometryClasses.h"

#include <math.h>

#include <src/Cuda/Common/JacobianCuda.h>
#include <src/Cuda/Common/UtilsCuda.h>
#include <src/Cuda/Common/LinearAlgebraCuda.h>
#include <src/Cuda/Common/TransformCuda.h>

#include <Open3D/Odometry/OdometryOption.h>

#include <src/Cuda/Camera/PinholeCameraIntrinsicCuda.h>
#include <src/Cuda/Container/ArrayCuda.h>

#include <Cuda/Geometry/RGBDImageCuda.h>
#include <Cuda/Geometry/ImageCuda.h>
#include <Eigen/Eigen>

namespace open3d {

namespace cuda {
#define CHECK_ODOMETRY_INLIERS_
#define CHECK_ODOMETRY_CORRESPONDENCES_

/**
 * We assume that the
 * - depths are **converted** from short
 * - intensities are **converted** from BGR / RGB ... whatever.
 *
 * Refer to this paper:
 * http://vladlen.info/papers/colored-point-cloud-registration-supplement.pdf
 *
 * We minimize
 * E(\xi) =
 * \sum_{p}
 *   (1 - sigma) ||I_{target}[g(s(h(p, D_{source}), \xi))] - I_{source}[p]||^2
 * + sigma ||D_{target}[g(s(h(p, D_{source}), \xi))] - s(h(p, D_{source})).z||^2
 *
 * Usually @target frame should be a keyframe, or 'the previous' frame
 *                 it should hold more precomputed information,
 *                 including gradients.
 *         @source frame should be a current frame.
 *         We warp the @source frame to the @target frame.
 */

typedef ImageCudaDevice<float, 1> ImageCudaDevicef;
typedef ImageCuda<float, 1> ImageCudaf;

template<size_t N>
class RGBDOdometryCudaDevice {
public:
    RGBDImageCudaDevice source_input_;
    RGBDImageCudaDevice target_input_;

    ImageCudaDevicef source_depth_[N];
    ImageCudaDevicef source_intensity_[N];

    ImageCudaDevicef target_depth_[N];
    ImageCudaDevicef target_intensity_[N];

    ImageCudaDevicef target_depth_dx_[N];
    ImageCudaDevicef target_depth_dy_[N];

    ImageCudaDevicef target_intensity_dx_[N];
    ImageCudaDevicef target_intensity_dy_[N];

    ArrayCudaDevice<float> results_;
    ArrayCudaDevice<Vector4i> correspondences_;

public:
    PinholeCameraIntrinsicCuda intrinsics_[N];
    TransformCuda transform_source_to_target_;

public:
    /** (1-sigma) * JtJ_I + sigma * JtJ_D **/
    /** To compute JtJ, we use \sqrt(1-sigma) J_I and \sqrt(sigma) J_D **/
    float sigma_;
    float sqrt_coeff_I_;
    float sqrt_coeff_D_;

public:
    float min_depth_;
    float max_depth_;
    float max_depth_diff_;

public:
    __HOSTDEVICE__ inline bool IsValidDepth(float depth) {
        return
#ifdef __CUDACC__
            !isnan(depth)
#else
            !std::isnan(depth)
#endif
        && depth > 0 && min_depth_ <= depth && depth <= max_depth_;
    }

    __HOSTDEVICE__ inline bool IsValidDepthDiff(float depth_diff) {
        return fabsf(depth_diff) <= max_depth_diff_;
    }

public:
    __DEVICE__ bool ComputePixelwiseCorrespondence(
        int x_source, int y_source, size_t level,
        int &x_target, int &y_target,
        Vector3f &X_source_on_target, float &d_target);

    __DEVICE__ bool ComputePixelwiseJacobianAndResidual(
        int x_source, int y_source, int x_target, int y_target, size_t level,
        const Vector3f &X_source_on_target, const float &d_target,
        Vector6f &jacobian_I, Vector6f &jacobian_D,
        float &residual_I, float &residual_D);

    __DEVICE__ bool ComputePixelwiseCorrespondenceAndInformationJacobian(
        int x_source, int y_source, /* Always size 0 */
        Vector6f &jacobian_x, Vector6f &jacobian_y, Vector6f &jacobian_z);

public:
    friend class RGBDOdometryCuda<N>;
};

template<size_t N>
class RGBDOdometryCuda {
public:
    std::shared_ptr<RGBDOdometryCudaDevice<N>> device_ = nullptr;

public:
    RGBDImageCuda source_input_;
    RGBDImageCuda target_input_;

    /** Core in RGBD Odometry **/
    ImageCudaf source_depth_[N];
    ImageCudaf source_intensity_[N];

    ImageCudaf target_depth_[N];
    ImageCudaf target_intensity_[N];

    ImageCudaf target_depth_dx_[N];
    ImageCudaf target_depth_dy_[N];
    ImageCudaf target_intensity_dx_[N];
    ImageCudaf target_intensity_dy_[N];

    ArrayCuda<float> results_;

    /** Debug use **/
    ArrayCuda<Vector4i> correspondences_;

public:
    float sigma_;
    odometry::OdometryOption option_;
    camera::PinholeCameraIntrinsic intrinsics_;
    Eigen::Matrix4d transform_source_to_target_;

    /** At current I don't want to add assignments for such a large class **/
    /** Ideally Create and Release should be only called once **/
    RGBDOdometryCuda();
    ~RGBDOdometryCuda();

    void SetParameters(const odometry::OdometryOption &option,
                       float sigma = 0.5f);
    void SetIntrinsics(camera::PinholeCameraIntrinsic intrinsics);

    bool Create(int width, int height);
    void Release();
    void UpdateDevice();
    void UpdateSigma(float sigma);

    void Initialize(RGBDImageCuda &source, RGBDImageCuda &target);

    std::tuple<bool, Eigen::Matrix4d, float> DoSingleIteration(
        size_t level, int iter);
    void ExtractResults(std::vector<float> &results,
                        Eigen::Matrix6d &JtJ, Eigen::Vector6d &Jtr,
                        float &loss, float &inliers);

    std::tuple<bool, Eigen::Matrix4d, std::vector<std::vector<float>>>
    ComputeMultiScale();

    Eigen::Matrix6d ComputeInformationMatrix();
};

template<size_t N>
class RGBDOdometryCudaKernelCaller {
public:
    static void DoSingleIteration(RGBDOdometryCuda<N> &odometry, size_t level);
    static void ComputeInformationMatrix(RGBDOdometryCuda<N> &odometry);

    static void PreprocessInput(RGBDOdometryCuda<N> &odometry,
                                ImageCudaf &source_depth_preprocessed,
                                ImageCudaf &source_intensity_preprocessed,
                                ImageCudaf &target_depth_preprocessed,
                                ImageCudaf &target_intensity_preprocessed);

    static void NormalizeIntensity(RGBDOdometryCuda<N> &odometry);
};

template<size_t N>
__GLOBAL__
void DoSingleIterationKernel(RGBDOdometryCudaDevice<N> odometry, size_t level);

template<size_t N>
__GLOBAL__
void ComputeInformationMatrixKernel(RGBDOdometryCudaDevice<N> odometry);

template<size_t N>
__GLOBAL__
void PreprocessInputKernel(RGBDOdometryCudaDevice<N> odometry,
                           ImageCudaDevicef source_depth_preprocessed,
                           ImageCudaDevicef source_intensity_preprocessed,
                           ImageCudaDevicef target_depth_preprocessed,
                           ImageCudaDevicef target_intensity_preprocessed);

template<size_t N>
__GLOBAL__
void NormalizeIntensityKernel(RGBDOdometryCudaDevice<N> odometry,
                              ArrayCudaDevice<float> means);

template<size_t N>
__GLOBAL__
void ComputeInitCorrespondenceMeanKernel(RGBDOdometryCudaDevice<N> odometry,
                                         ArrayCudaDevice<float> means);

} // cuda
} // open3d