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

#include <Cuda/Common/VectorCuda.h>
#include <Cuda/Common/TransformCuda.h>
#include <Cuda/Geometry/ImagePyramidCuda.h>
#include <Cuda/Geometry/RGBDImagePyramidCuda.h>

#include <Eigen/Eigen>

namespace open3d {

#define CHECK_ODOMETRY_INLIERS_
#define CHECK_ODOMETRY_CORRESPONDENCES_

/**
 * In Inverse Compositional (IC) version of RGBDOdometry, instead of warping
 * from source to target, we warp target to source
 *
 * I_{source}[w(0 + \delta xi)] - I_{target}[w(p)]
 *
 * e.g., see
 * http://homepages.inf.ed.ac.uk/rbf/CVonline/LOCAL_COPIES/AV0910/zhao.pdf
 */
template<size_t N>
class ICRGBDOdometryCudaServer {
private:
    ImagePyramidCudaServer<Vector1f, N> source_on_target_;

    RGBDImagePyramidCudaServer<N> source_;
    RGBDImagePyramidCudaServer<N> source_dx_;
    RGBDImagePyramidCudaServer<N> source_dy_;

    RGBDImagePyramidCudaServer<N> target_;

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
    __DEVICE__ bool ComputePixelwiseJacobianAndResidual(
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
    __HOSTDEVICE__ inline RGBDImagePyramidCudaServer<N> &source_dx() {
        return source_dx_;
    }
    __HOSTDEVICE__ inline RGBDImagePyramidCudaServer<N>& source_dy() {
        return source_dy_;
    }

    __HOSTDEVICE__ inline RGBDImagePyramidCudaServer<N> &target() {
        return target_;
    }

    __HOSTDEVICE__ inline ArrayCudaServer<float> &results() {
        return results_;
    }
    __HOSTDEVICE__ inline ArrayCudaServer<Vector4i> &correspondences() {
        return correspondences_;
    }

    friend class ICRGBDOdometryCuda<N>;
};

template<size_t N>
class ICRGBDOdometryCuda {
private:
    std::shared_ptr<ICRGBDOdometryCudaServer<N>> server_ = nullptr;

    ImagePyramidCuda<Vector1f, N> source_on_target_;
    RGBDImagePyramidCuda<N> source_raw_;
    RGBDImagePyramidCuda<N> target_raw_;

    RGBDImagePyramidCuda<N> source_;
    RGBDImagePyramidCuda<N> target_;
    RGBDImagePyramidCuda<N> source_dx_;
    RGBDImagePyramidCuda<N> source_dy_;

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
    ICRGBDOdometryCuda();
    ~ICRGBDOdometryCuda();

    void SetParameters(const OdometryOption &option, const float sigma = 0.5f);
    void SetIntrinsics(PinholeCameraIntrinsic intrinsics);

    bool Create(int width, int height);
    void Release();
    void UpdateServer();

    void PrepareData(RGBDImageCuda &source, RGBDImageCuda &target);
    void ExtractResults(std::vector<float> &results,
                        EigenMatrix6d &JtJ, EigenVector6d &Jtr,
                        float &loss, float &inliers);

    std::tuple<bool, Eigen::Matrix4d, float>
    DoSingleIteration(size_t level, int iter);
    std::tuple<bool, Eigen::Matrix4d, std::vector<std::vector<float>>>
    ComputeMultiScale();

    RGBDImagePyramidCuda<N> &source() { return source_; }
    RGBDImagePyramidCuda<N> &target() { return target_; }

    std::shared_ptr<ICRGBDOdometryCudaServer<N>> &server() {
        return server_;
    }
    const std::shared_ptr<ICRGBDOdometryCudaServer<N>> &server() const {
        return server_;
    }
};

template<size_t N>
class ICRGBDOdometryCudaKernelCaller {
public:
    static __HOST__ void ApplyICRGBDOdometryKernelCaller(
        ICRGBDOdometryCudaServer<N>&server, size_t level,
        int width, int height);
};

template<size_t N>
__GLOBAL__
void ApplyICRGBDOdometryKernel(ICRGBDOdometryCudaServer<N> odometry, size_t level);

}
