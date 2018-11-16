//
// Created by wei on 11/9/18.
//

#pragma once

#include "SequentialRGBDOdometryCuda.h"
#include <sophus/se3.hpp>

namespace open3d {
/**
 * Client end
 * TODO: Think about how do we use server_ ... we don't want copy
 * constructors for such a large system...
 */
template<size_t N>
RGBDOdometryCuda<N>::RGBDOdometryCuda() : server_(nullptr) {}

template<size_t N>
RGBDOdometryCuda<N>::~RGBDOdometryCuda() {
    Release();
}

template<size_t N>
void RGBDOdometryCuda<N>::SetParameters(float sigma,
                                        float depth_near_threshold,
                                        float depth_far_threshold,
                                        float depth_diff_threshold) {
    sigma_ = sigma;
    depth_near_threshold_ = depth_near_threshold;
    depth_far_threshold_ = depth_far_threshold;
    depth_diff_threshold_ = depth_diff_threshold;
}

template<size_t N>
void RGBDOdometryCuda<N>::SetIntrinsics(PinholeCameraIntrinsic intrinsics) {
    intrinsics_ = intrinsics;
}

template<size_t N>
bool RGBDOdometryCuda<N>::Create(int width, int height) {
    assert(width > 0 && height > 0);

    if (server_ != nullptr) {
        if (source_[0].width_ != width || source_[0].height_ != height) {
            PrintError("[RGBDOdometryCuda] Incompatible image size, "
                       "width: %d vs %d, height: %d vs %d, "
                       "@Create aborted.\n",
                       source_[0].width_, width, source_[0].height_, height);
            return false;
        }
        return true;
    }

    server_ = std::make_shared<RGBDOdometryCudaServer<N>>();

    source_on_target_.Create(width, height);

    source_.Create(width, height);
    target_.Create(width, height);
    target_dx_.Create(width, height);
    target_dy_.Create(width, height);

    results_.Create(29);

    UpdateServer();
    return true;
}

template<size_t N>
void RGBDOdometryCuda<N>::Release() {
    source_.Release();
    target_.Release();
    target_dx_.Release();
    target_dy_.Release();

    source_on_target_.Release();

    results_.Release();

    server_ = nullptr;
}

template<size_t N>
void RGBDOdometryCuda<N>::UpdateServer() {
    if (server_ != nullptr) {
        source_on_target_.UpdateServer();
        server_->source_on_target() = *source_on_target_.server();

        source_.UpdateServer();
        server_->source() = *source_.server();

        target_.UpdateServer();
        server_->target() = *target_.server();

        target_dx_.UpdateServer();
        server_->target_dx() = *target_dx_.server();

        target_dy_.UpdateServer();
        server_->target_dy() = *target_dy_.server();

        server_->results() = *results_.server();

        /** Update parameters **/
        server_->sigma_ = sigma_;
        server_->sqrt_coeff_D_ = sqrtf(sigma_);
        server_->sqrt_coeff_I_ = sqrtf(1 - sigma_);
        server_->depth_near_threshold_ = depth_near_threshold_;
        server_->depth_far_threshold_ = depth_far_threshold_;
        server_->depth_diff_threshold_ = depth_diff_threshold_;

        server_->intrinsics_[0] = PinholeCameraIntrinsicCuda(intrinsics_);
        for (size_t i = 1; i < N; ++i) {
            server_->intrinsics_[i] = server_->intrinsics_[i - 1].Downsample();
        }
    }
}

template<size_t N>
void RGBDOdometryCuda<N>::ExtractResults(std::vector<float> &results,
                                         EigenMatrix6d &JtJ,
                                         EigenVector6d &Jtr,
                                         float &loss, float &inliers) {
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
    loss = results[cnt];
    ++cnt;
    inliers = results[cnt];
}

template<size_t N>
void RGBDOdometryCuda<N>::PrepareData(
    RGBDImageCuda &source, RGBDImageCuda &target) {
    assert(source.width_ == target.width_);
    assert(source.height_ == target.height_);

    bool success = Create(source.width_, source.height_);
    if (! success) {
        PrintError("[RGBDOdometryCuda] create failed, "
                   "@PrepareData aborted.\n");
        return;
    }

    source_.Build(source);
    target_.Build(target);
    for (size_t i = 0; i < N; ++i) {
        target_[i].depthf().Sobel(target_dx_[i].depthf(),
                                  target_dy_[i].depthf(),
                                  false);
        target_[i].intensity().Sobel(target_dx_[i].intensity(),
                                     target_dy_[i].intensity(),
                                     false);
    }
    UpdateServer();
}

template<size_t N>
float RGBDOdometryCuda<N>::ApplyOneIterationOnLevel(size_t level, int iter) {
    results_.Memset(0);

#ifdef VISUALIZE_ODOMETRY_INLIERS
    source_on_target_[level].CopyFrom(target_[level].intensity());
#endif
    server_->transform_source_to_target_.FromEigen(
        transform_source_to_target_);

    RGBDOdometryCudaKernelCaller<N>::ApplyRGBDOdometryKernelCaller(
        *server_, level,
        source_[level].depthf().width_,
        source_[level].depthf().height_);

#ifdef VISUALIZE_ODOMETRY_INLIERS
    cv::Mat im = source_on_target_[level].DownloadMat();
        cv::imshow("source_on_target", im);
        cv::waitKey(-1);
#endif

    std::vector<float> results = results_.DownloadAll();

    EigenMatrix6d JtJ;
    EigenVector6d Jtr;
    float loss, inliers;
    ExtractResults(results, JtJ, Jtr, loss, inliers);

    PrintDebug("> Level %d, iter %d: loss = %f, avg loss = %f, "
               "inliers = %.0f\n",
               level, iter, loss, loss / inliers, inliers);

    EigenVector6d dxi = JtJ.ldlt().solve(-Jtr);
    transform_source_to_target_ =
        Sophus::SE3d::exp(dxi).matrix() * transform_source_to_target_;

    return loss;
}

template<size_t N>
void RGBDOdometryCuda<N>::Apply() {

    const int kIterations[] = {3, 15, 60};
    for (int level = (int) (N - 1); level >= 0; --level) {
        for (int iter = 0; iter < kIterations[level]; ++iter) {
            ApplyOneIterationOnLevel((size_t) level, iter);
        }
    }
}
}
