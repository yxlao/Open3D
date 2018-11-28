//
// Created by wei on 10/1/18.
//

#pragma once

#include "OdometryClasses.h"
#include "JacobianCuda.h"

#include <Cuda/Common/UtilsCuda.h>

#include <Cuda/Camera/PinholeCameraIntrinsicCuda.h>
#include <Cuda/Container/ArrayCuda.h>

#include <Cuda/Common/VectorCuda.h>
#include <Cuda/Common/TransformCuda.h>
#include <Cuda/Geometry/ImagePyramidCuda.h>
#include <Cuda/Geometry/RGBDImagePyramidCuda.h>

#include <Eigen/Eigen>

namespace open3d {
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
class RGBDOdometryCudaServer {
private:
    ImagePyramidCudaServer<Vector1f, N> source_on_target_;

    RGBDImagePyramidCudaServer<N> source_;
    RGBDImagePyramidCudaServer<N> target_;
    RGBDImagePyramidCudaServer<N> target_dx_;
    RGBDImagePyramidCudaServer<N> target_dy_;

    ArrayCudaServer<float> results_;
    ArrayCudaServer<Vector4i> correspondences_;

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
    __DEVICE__ bool ComputePixelwiseJacobiansAndResiduals(
        int x, int y, size_t level,
        JacobianCuda<6> &jacobian_I, JacobianCuda<6> &jacobian_D,
        float &residual_I, float &residual_D);
    __DEVICE__ bool ComputePixelwiseJtJAndJtr(
        JacobianCuda<6> &jacobian_I, JacobianCuda<6> &jacobian_D,
        float &residual_I, float &residual_D,
        HessianCuda<6> &JtJ, Vector6f &Jtr);

public:
    __HOSTDEVICE__ inline ImagePyramidCudaServer<Vector1f, N> &
    source_on_target() {
        return source_on_target_;
    }
    __HOSTDEVICE__ inline RGBDImagePyramidCudaServer<N> &source() {
        return source_;
    }
    __HOSTDEVICE__ inline RGBDImagePyramidCudaServer<N> &target() {
        return target_;
    }
    __HOSTDEVICE__ inline RGBDImagePyramidCudaServer<N> &target_dx() {
        return target_dx_;
    }
    __HOSTDEVICE__ inline RGBDImagePyramidCudaServer<N>& target_dy() {
        return target_dy_;
    }

    __HOSTDEVICE__ inline ArrayCudaServer<float> &results() {
        return results_;
    }
    __HOSTDEVICE__ inline ArrayCudaServer<Vector4i> &correspondences() {
        return correspondences_;
    }

    friend class RGBDOdometryCuda<N>;
};

template<size_t N>
class RGBDOdometryCuda {
private:
    std::shared_ptr<RGBDOdometryCudaServer<N>> server_ = nullptr;

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
    float depth_near_threshold_;
    float depth_far_threshold_;
    float depth_diff_threshold_;

    PinholeCameraIntrinsic intrinsics_;
    Eigen::Matrix4d transform_source_to_target_;

    /** At current I don't want to add assignments for such a large class **/
    /** Ideally Create and Release should be only called once **/
    RGBDOdometryCuda();
    ~RGBDOdometryCuda();
    void SetParameters(float sigma,
                       float depth_near_threshold, float depth_far_threshold,
                       float depth_diff_threshold);
    void SetIntrinsics(PinholeCameraIntrinsic intrinsics);

    bool Create(int width, int height);
    void Release();
    void UpdateServer();

    void PrepareData(RGBDImageCuda &source, RGBDImageCuda &target);

    float ApplyOneIterationOnLevel(size_t level, int iter);
    void Apply();

    void ExtractResults(std::vector<float> &results,
                        EigenMatrix6d &JtJ, EigenVector6d &Jtr,
                        float &loss, float &inliers);

    RGBDImagePyramidCuda<N> &source() { return source_; }
    RGBDImagePyramidCuda<N> &target() { return target_; }

    std::shared_ptr<RGBDOdometryCudaServer<N>> &server() {
        return server_;
    }
    const std::shared_ptr<RGBDOdometryCudaServer<N>> &server() const {
        return server_;
    }
};

template<size_t N>
class RGBDOdometryCudaKernelCaller {
public:
    static __HOST__ void ApplyRGBDOdometryKernelCaller(
        RGBDOdometryCudaServer<N>&server, size_t level,
        int width, int height);
};

template<size_t N>
__GLOBAL__
void ApplyRGBDOdometryKernel(RGBDOdometryCudaServer<N> odometry, size_t level);

}
