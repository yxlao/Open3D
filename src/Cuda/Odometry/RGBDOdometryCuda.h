//
// Created by wei on 10/1/18.
//

#pragma once

#include "OdometryClasses.h"
#include "JacobianCuda.h"

#include <Core/Odometry/OdometryOption.h>

#include <Cuda/Common/UtilsCuda.h>

#include <Cuda/Camera/PinholeCameraIntrinsicCuda.h>
#include <Cuda/Container/ArrayCuda.h>

#include <Cuda/Common/LinearAlgebraCuda.h>
#include <Cuda/Common/TransformCuda.h>
#include <Cuda/Geometry/ImagePyramidCuda.h>
#include <Cuda/Geometry/RGBDImagePyramidCuda.h>

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
template<size_t N>
class RGBDOdometryCudaDevice {
public:
    ImagePyramidCudaDevice<Vector1f, N> source_on_target_;

    RGBDImagePyramidCudaDevice<N> source_;
    RGBDImagePyramidCudaDevice<N> target_;
    RGBDImagePyramidCudaDevice<N> target_dx_;
    RGBDImagePyramidCudaDevice<N> target_dy_;

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
    float depth_near_threshold_;
    float depth_far_threshold_;
    float depth_diff_threshold_;

public:
    __HOSTDEVICE__ inline bool IsValidDepth(float depth) {
        return depth_near_threshold_ <= depth && depth <= depth_far_threshold_;
    }
    __HOSTDEVICE__ inline bool IsValidDepthDiff(float depth_diff) {
        return fabsf(depth_diff) <= depth_diff_threshold_;
    }

public:
    __DEVICE__ bool ComputePixelwiseCorrespondenceAndResidual(
        int x_source, int y_source, size_t level,
        int &x_target, int &y_target,
        Vector3f &X_target,
        float &residual_I, float &residual_D);

    __DEVICE__ bool ComputePixelwiseJacobian(
        int x_target, int y_target, size_t level,
        const Vector3f &X_target,
        Vector6f &jacobian_I, Vector6f &jacobian_D);

    __DEVICE__ void ComputePixelwiseJtJAndJtr(
        const Vector6f &jacobian_I, const Vector6f &jacobian_D,
        const float &residual_I, const float &residual_D,
        HessianCuda<6> &JtJ, Vector6f &Jtr);

public:
    friend class RGBDOdometryCuda<N>;
};

template<size_t N>
class RGBDOdometryCuda {
private:
    std::shared_ptr<RGBDOdometryCudaDevice<N>> server_ = nullptr;

    ImagePyramidCuda<Vector1f, N> source_on_target_;
    RGBDImagePyramidCuda<N> source_raw_;
    RGBDImagePyramidCuda<N> target_raw_;

    RGBDImagePyramidCuda<N> source_;
    RGBDImagePyramidCuda<N> target_;
    RGBDImagePyramidCuda<N> target_dx_;
    RGBDImagePyramidCuda<N> target_dy_;

    ArrayCuda<float> results_;

public:
    ArrayCuda<Vector4i> correspondences_;

public:
    typedef Eigen::Matrix<double, 6, 6> EigenMatrix6d;
    typedef Eigen::Matrix<double, 6, 1> EigenVector6d;

    float sigma_;
    OdometryOption option_;
    PinholeCameraIntrinsic intrinsics_;
    Eigen::Matrix4d transform_source_to_target_;

    /** At current I don't want to add assignments for such a large class **/
    /** Ideally Create and Release should be only called once **/
    RGBDOdometryCuda();
    ~RGBDOdometryCuda();

    void SetParameters(const OdometryOption &option, const float sigma = 0.5f);
    void SetIntrinsics(PinholeCameraIntrinsic intrinsics);

    bool Create(int width, int height);
    void Release();
    void UpdateServer();

    void Initialize(RGBDImageCuda &source, RGBDImageCuda &target);

    std::tuple<bool, Eigen::Matrix4d, float> DoSingleIteration(
        size_t level, int iter);
    void ExtractResults(
        std::vector<float> &results,
        EigenMatrix6d &JtJ, EigenVector6d &Jtr, float &loss, float &inliers);

    std::tuple<bool, Eigen::Matrix4d, std::vector<std::vector<float>>>
    ComputeMultiScale();

    RGBDImagePyramidCuda<N> &source() { return source_; }
    RGBDImagePyramidCuda<N> &target() { return target_; }

    std::shared_ptr<RGBDOdometryCudaDevice<N>> &server() {
        return server_;
    }
    const std::shared_ptr<RGBDOdometryCudaDevice<N>> &server() const {
        return server_;
    }
};

template<size_t N>
class RGBDOdometryCudaKernelCaller {
public:
    static __HOST__ void DoSingleIterationKernelCaller(
        RGBDOdometryCudaDevice<N> &server, size_t level,
        int width, int height);
};

template<size_t N>
__GLOBAL__
void DoSingleIterationKernel(RGBDOdometryCudaDevice<N> odometry, size_t level);

} // cuda
} // open3d