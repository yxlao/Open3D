//
// Created by wei on 10/1/18.
//

#pragma once

#include "RGBDOdometryCuda.h"
#include <Cuda/Geometry/ImageCudaDevice.cuh>
#include <Cuda/Container/ArrayCudaDevice.cuh>
#include <sophus/se3.hpp>

namespace open3d {

/**
 * Server end
 */
template<size_t N>
__device__
bool RGBDOdometryCudaServer<N>::ComputePixelwiseJacobiansAndResiduals(
    int x, int y, size_t level,
    JacobianCuda<6> &jacobian_I,
    JacobianCuda<6> &jacobian_D,
    float &residual_I,
    float &residual_D) {

    /** Check 1: depth valid in source? **/
    float d_source = source_depth()[level].at(x, y)(0);
    bool mask = IsValidDepth(d_source);
    if (!mask) return false;

    /** Check 2: reprojected point in image? **/
    Vector3f
        X = transform_source_to_target_
        * intrinsics_[level].InverseProjectPixel(
            Vector2i(x, y), d_source);

    Vector2f p_warped = intrinsics_[level].ProjectPoint(X);
    mask = intrinsics_[level].IsPixelValid(p_warped);
    if (!mask) return false;

    /** Check 3: depth valid in target? Occlusion? -> 1ms **/
    float d_target = target_depth()[level].interp_with_holes_at(
        p_warped(0), p_warped(1))(0);
    mask = IsValidDepth(d_target) && IsValidDepthDiff(d_target - X(2));
    if (!mask) return false;

    /** Checks passed, let's rock! -> 3ms, can be 2ms faster if we don't use
     * interpolation
     *  \partial D(p_warped) \partial p_warped: [dx_D, dy_D] at p_warped, 1x2
     *  \partial I(p_warped) \partial p_warped: [dx_I, dy_I] at p_warped, 1x2
     *  \partial X.z \partial X: [0, 0, 1], 1x3
     *  \partial p_warped \partial X: [fx/Z, 0, -fx X/Z^2;
     *                                 0, fy/Z, -fy Y/Z^2]            2x3
     *  \partial X \partial \xi: [I | -[X]^] = [1 0 0 0  Z -Y;
     *                                          0 1 0 -Z 0 X;
     *                                          0 0 1 Y -X 0]         3x6
     * J_I = (d I(p_warped) / d p_warped) (d p_warped / d X) (d X / d \xi)
     * J_D = (d D(p_warped) / d p_warped) (d p_warped / d X) (d X / d \xi)
     *     - (d X.z / d X) (d X / d \xi)
     */
    float dx_I = target_intensity_dx()[level].interp_at(
        p_warped(0), p_warped(1))(0);
    float dy_I = target_intensity_dy()[level].interp_at(
        p_warped(0), p_warped(1))(0);
    float dx_D = target_depth_dx()[level].interp_at(
        p_warped(0), p_warped(1))(0);
    float dy_D = target_depth_dy()[level].interp_at(
        p_warped(0), p_warped(1))(0);
    float fx = intrinsics_[level].fx_;
    float fy = intrinsics_[level].fy_;
    float inv_Z = 1.0f / X(2);
    float fx_on_Z = fx * inv_Z;
    float fy_on_Z = fy * inv_Z;

    float c0 = dx_I * fx_on_Z;
    float c1 = dy_I * fy_on_Z;
    float c2 = -(c0 * X(0) + c1 * X(1)) * inv_Z;
    jacobian_I(0) = sqrt_coeff_I_ * c0;
    jacobian_I(1) = sqrt_coeff_I_ * c1;
    jacobian_I(2) = sqrt_coeff_I_ * c2;
    jacobian_I(3) = sqrt_coeff_I_ * (-X(2) * c1 + X(1) * c2);
    jacobian_I(4) = sqrt_coeff_I_ * (X(2) * c0 - X(0) * c2);
    jacobian_I(5) = sqrt_coeff_I_ * (-X(1) * c0 + X(0) * c1);
    residual_I = sqrt_coeff_I_ *
        (target_intensity()[level].interp_at(p_warped(0), p_warped(1))(0)
            - source_intensity()[level].at(x, y)(0));

    float d0 = dx_D * fx_on_Z;
    float d1 = dy_D * fy_on_Z;
    float d2 = -(d0 * X(0) + d1 * X(1)) * inv_Z;
    jacobian_D(0) = sqrt_coeff_D_ * d0;
    jacobian_D(1) = sqrt_coeff_D_ * d1;
    jacobian_D(2) = sqrt_coeff_D_ * (d2 - 1.0f);
    jacobian_D(3) = sqrt_coeff_D_ * ((-X(2) * d1 + X(1) * d2) - X(1));
    jacobian_D(4) = sqrt_coeff_D_ * ((X(2) * d0 - X(0) * d2) + X(0));
    jacobian_D(5) = sqrt_coeff_D_ * (-X(1) * d0 + X(0) * d1);
    residual_D = sqrt_coeff_D_ * (d_target - X(2));

#ifdef VISUALIZE_ODOMETRY_INLIERS
    source_on_target()[level].at((int) p_warped(0), (int) p_warped(1))
        = Vector1f(0.0f);
#endif
    return true;
}

template<size_t N>
__device__
bool RGBDOdometryCudaServer<N>::ComputePixelwiseJtJAndJtr(
    JacobianCuda<6> &jacobian_I, JacobianCuda<6> &jacobian_D,
    float &residual_I, float &residual_D,
    HessianCuda<6> &JtJ, Vector6f &Jtr) {

    int cnt = 0;
#pragma unroll 1
    for (int i = 0; i < 6; ++i) {
#pragma unroll 1
        for (int j = i; j < 6; ++j) {
            JtJ(cnt++) = jacobian_I(i) * jacobian_I(j)
                + jacobian_D(i) * jacobian_D(j);
        }
        Jtr(i) = jacobian_I(i) * residual_I + jacobian_D(i) * residual_D;
    }

    return true;
}

/**
 * Client end
 * TODO: Think about how do we use server_ ... we don't want copy
 * constructors for such a large system...
 */
template<size_t N>
RGBDOdometryCuda<N>::RGBDOdometryCuda() {
    server_ = std::make_shared<RGBDOdometryCudaServer<N>>();
}

template<size_t N>
RGBDOdometryCuda<N>::~RGBDOdometryCuda() {
    Release();
    server_ = nullptr;
}

template<size_t N>
void RGBDOdometryCuda<N>::SetParameters(float sigma,
                                        float depth_near_threshold,
                                        float depth_far_threshold,
                                        float depth_diff_threshold) {
    server_->sigma_ = sigma;
    server_->sqrt_coeff_D_ = sqrtf(sigma);
    server_->sqrt_coeff_I_ = sqrtf(1 - sigma);
    server_->depth_diff_threshold_ = depth_near_threshold;
    server_->depth_far_threshold_ = depth_far_threshold;
    server_->depth_diff_threshold_ = depth_diff_threshold;
}

template<size_t N>
void RGBDOdometryCuda<N>::SetIntrinsics(PinholeCameraIntrinsic intrinsics) {
    intrinsics_ = intrinsics;
    server_->intrinsics_[0] = PinholeCameraIntrinsicCuda(intrinsics);
    for (size_t i = 1; i < N; ++i) {
        server_->intrinsics_[i] = server_->intrinsics_[i - 1].Downsample();
    }
}

template<size_t N>
void RGBDOdometryCuda<N>::Create(int width, int height) {
    if (server_ != nullptr) {
        PrintError("[RGBDOdometryCuda] Already created, abort!\n");
        return;
    }

    target_depth_.Create(width, height);
    target_depth_dx_.Create(width, height);
    target_depth_dy_.Create(width, height);

    target_intensity_.Create(width, height);
    target_intensity_dx_.Create(width, height);
    target_intensity_dy_.Create(width, height);

    source_depth_.Create(width, height);
    source_intensity_.Create(width, height);

    source_on_target_.Create(width, height);

    results_.Create(29);

    UpdateServer();
}

template<size_t N>
void RGBDOdometryCuda<N>::Release() {
    target_depth_.Release();
    target_depth_dx_.Release();
    target_depth_dy_.Release();

    target_intensity_.Release();
    target_intensity_dx_.Release();
    target_intensity_dy_.Release();

    source_depth_.Release();
    source_intensity_.Release();

    source_on_target_.Release();

    results_.Release();
}

template<size_t N>
void RGBDOdometryCuda<N>::UpdateServer() {
    if (server_ != nullptr) {
        server_->target_depth() = *target_depth_.server();
        server_->target_depth_dx() = *target_depth_dx_.server();
        server_->target_depth_dy() = *target_depth_dy_.server();

        server_->target_intensity() = *target_intensity_.server();
        server_->target_intensity_dx() = *target_intensity_dx_.server();
        server_->target_intensity_dy() = *target_intensity_dy_.server();

        server_->source_depth() = *source_depth_.server();
        server_->source_intensity() = *source_intensity_.server();

        server_->source_on_target() = *source_on_target_.server();

        server_->results() = *results_.server();
    }
}

template<size_t N>
void RGBDOdometryCuda<N>::ExtractResults(std::vector<float> &results,
                                         EigenMatrix6d &JtJ,
                                         EigenVector6d &Jtr,
                                         float &error, float &inliers) {
    int cnt = 0;
    for (int i = 0; i < 6; ++i) {
        for (int j = i; j < 6; ++j) {
            JtJ(i, j) = JtJ(j, i) = results[cnt];
            ++cnt;
        }
    }
    for (int i = 0; i < 6; ++i) {
        Jtr(i) = results[cnt];
        ++cnt;
    }
    error = results[cnt];
    ++cnt;
    inliers = results[cnt];
}

template<size_t N>
void RGBDOdometryCuda<N>::Build(ImageCuda<Vector1f> &source_depth,
                                ImageCuda<Vector1f> &source_intensity,
                                ImageCuda<Vector1f> &target_depth,
                                ImageCuda<Vector1f> &target_intensity) {
    assert(source_depth.width_ == source_intensity.width_);
    assert(source_depth.height_ == source_intensity.height_);
    assert(target_depth.width_ == target_intensity.width_);
    assert(target_depth.height_ == target_intensity.height_);

    int source_width = source_depth.width_;
    int source_height = source_depth.height_;
    int target_width = target_depth.width_;
    int target_height = target_depth.height_;
    assert(source_width > 0 && source_height > 0);
    assert(target_width > 0 && target_height > 0);

    source_depth_.Build(source_depth);
    source_intensity_.Build(source_intensity);
    target_depth_.Build(target_depth);
    target_intensity_.Build(target_intensity);

    target_depth_dx_.Create(target_width, target_height);
    target_depth_dy_.Create(target_width, target_height);
    target_intensity_dx_.Create(target_width, target_height);
    target_intensity_dy_.Create(target_width, target_height);

    source_on_target_.Create(source_width, source_height);

    for (size_t i = 0; i < N; ++i) {
        target_depth_[i].Sobel(target_depth_dx_[i],
                               target_depth_dy_[i],
                               false);
        target_intensity_[i].Sobel(target_intensity_dx_[i],
                                   target_intensity_dy_[i],
                                   false);
        source_on_target_[i].CopyFrom(target_intensity_[i]);
    }
    target_depth_dx_.UpdateServer();
    target_depth_dy_.UpdateServer();
    target_intensity_dx_.UpdateServer();
    target_intensity_dy_.UpdateServer();
    source_on_target_.UpdateServer();

    results_.Create(29);
    UpdateServer();
}

template<size_t N>
void RGBDOdometryCuda<N>::Apply(ImageCuda<Vector1f> &source_depth,
                                ImageCuda<Vector1f> &source_intensity,
                                ImageCuda<Vector1f> &target_depth,
                                ImageCuda<Vector1f> &target_intensity) {

    const int kIterations[] = {3, 5, 10};
    for (int level = (int) (N - 1); level >= 0; --level) {
        for (int iter = 0; iter < kIterations[level]; ++iter) {
            results_.Memset(0);

#ifdef VISUALIZE_ODOMETRY_INLIERS
            source_on_target_[level].CopyFrom(target_intensity_[level]);
#endif
            server_->transform_source_to_target_.FromEigen(
                transform_source_to_target_);

            const dim3 blocks(
                DIV_CEILING(source_depth_[level].width_, THREAD_2D_UNIT),
                DIV_CEILING(source_depth_[level].height_, THREAD_2D_UNIT));
            const dim3 threads(THREAD_2D_UNIT, THREAD_2D_UNIT);
            ApplyRGBDOdometryKernel << < blocks, threads >> > (*server_, level);
            CheckCuda(cudaDeviceSynchronize());
            CheckCuda(cudaGetLastError());

#ifdef VISUALIZE_ODOMETRY_INLIERS
            cv::Mat im = source_on_target_[level].DownloadMat();
            cv::imshow("source_on_target", im);
            cv::waitKey(-1);
#endif

            std::vector<float> results = results_.DownloadAll();

            EigenMatrix6d JtJ;
            EigenVector6d Jtr;
            float error, inliers;
            ExtractResults(results, JtJ, Jtr, error, inliers);

            PrintInfo("> Level %d, iter %d: error = %f, avg_error = %f, "
                      "inliers = %.0f\n",
                      level, iter, error, error / inliers, inliers);

            EigenVector6d dxi = JtJ.ldlt().solve(-Jtr);
            transform_source_to_target_ =
                Sophus::SE3d::exp(dxi).matrix() * transform_source_to_target_;

        }
    }
}
}
