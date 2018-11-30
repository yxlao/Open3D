//
// Created by wei on 11/9/18.
//

#pragma once

#include "ICRGBDOdometryCuda.h"
#include <sophus/se3.hpp>

namespace open3d {
/**
 * Client end
 * TODO: Think about how do we use server_ ... we don't want copy
 * constructors for such a large system...
 */
template<size_t N>
ICRGBDOdometryCuda<N>::ICRGBDOdometryCuda() : server_(nullptr) {}

template<size_t N>
ICRGBDOdometryCuda<N>::~ICRGBDOdometryCuda() {
    Release();
}

template<size_t N>
void ICRGBDOdometryCuda<N>::SetParameters(
    const OdometryOption &option, const float sigma) {
    assert(option_.iteration_number_per_pyramid_level_.size() == N);
    option_ = option;
    sigma_ = sigma;
}

template<size_t N>
void ICRGBDOdometryCuda<N>::SetIntrinsics(PinholeCameraIntrinsic intrinsics) {
    intrinsics_ = intrinsics;
}

template<size_t N>
bool ICRGBDOdometryCuda<N>::Create(int width, int height) {
    assert(width > 0 && height > 0);

    if (server_ != nullptr) {
        if (source_[0].width_ != width || source_[0].height_ != height) {
            PrintError("[ICRGBDOdometryCuda] Incompatible image size, "
                       "width: %d vs %d, height: %d vs %d, "
                       "@Create aborted.\n",
                       source_[0].width_, width, source_[0].height_, height);
            return false;
        }
        return true;
    }

    server_ = std::make_shared<ICRGBDOdometryCudaServer<N>>();

    source_on_target_.Create(width, height);

    source_.Create(width, height);
    source_dx_.Create(width, height);
    source_dy_.Create(width, height);
    target_.Create(width, height);

    results_.Create(29); // 21 + 6 + 2
    correspondences_.Create(width * height);

    UpdateServer();
    return true;
}

template<size_t N>
void ICRGBDOdometryCuda<N>::Release() {
    source_.Release();
    source_dx_.Release();
    source_dy_.Release();

    target_.Release();
    source_on_target_.Release();

    results_.Release();
    correspondences_.Release();

    server_ = nullptr;
}

template<size_t N>
void ICRGBDOdometryCuda<N>::UpdateServer() {
    if (server_ != nullptr) {
        source_on_target_.UpdateServer();
        server_->source_on_target() = *source_on_target_.server();

        source_.UpdateServer();
        server_->source() = *source_.server();

        target_.UpdateServer();
        server_->target() = *target_.server();

        source_dx_.UpdateServer();
        server_->source_dx() = *source_dx_.server();

        source_dy_.UpdateServer();
        server_->source_dy() = *source_dy_.server();

        server_->results() = *results_.server();
        server_->correspondences() = *correspondences_.server();

        /** Update parameters **/
        server_->sigma_ = sigma_;
        server_->sqrt_coeff_D_ = sqrtf(sigma_);
        server_->sqrt_coeff_I_ = sqrtf(1 - sigma_);
        server_->depth_near_threshold_ = (float) option_.min_depth_;
        server_->depth_far_threshold_ = (float) option_.max_depth_;
        server_->depth_diff_threshold_ = (float) option_.max_depth_diff_;

        server_->intrinsics_[0] = PinholeCameraIntrinsicCuda(intrinsics_);
        for (size_t i = 1; i < N; ++i) {
            server_->intrinsics_[i] = server_->intrinsics_[i - 1].Downsample();
        }
    }
}

template<size_t N>
void ICRGBDOdometryCuda<N>::ExtractResults(std::vector<float> &results,
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
void ICRGBDOdometryCuda<N>::PrepareData(
    RGBDImageCuda &source, RGBDImageCuda &target) {
    assert(source.width_ == target.width_);
    assert(source.height_ == target.height_);

    bool success = Create(source.width_, source.height_);
    if (!success) {
        PrintError("[ICRGBDOdometryCuda] create failed, "
                   "@PrepareData aborted.\n");
        return;
    }

    source_raw_.Build(source);
    target_raw_.Build(target);
    for (size_t i = 0; i < N; ++i) {
        /* Filter raw data */
        source_raw_[i].depthf().Gaussian(
            source_[i].depthf(), Gaussian3x3, true);
        source_raw_[i].intensity().Gaussian(
            source_[i].intensity(), Gaussian3x3, false);

        target_raw_[i].depthf().Gaussian(
            target_[i].depthf(), Gaussian3x3, true);
        target_raw_[i].intensity().Gaussian(
            target_[i].intensity(), Gaussian3x3, false);

        /* Compute gradients */
        source_[i].depthf().Sobel(
            source_dx_[i].depthf(), source_dy_[i].depthf(), true);
        source_[i].intensity().Sobel(
            source_dx_[i].intensity(), source_dy_[i].intensity(), false);
    }

    UpdateServer();
}

template<size_t N>
std::tuple<bool, Eigen::Matrix4d, float>
ICRGBDOdometryCuda<N>::DoSingleIteration(size_t level, int iter) {
    results_.Memset(0);
    correspondences_.set_iterator(0);

#ifdef VISUALIZE_ODOMETRY_INLIERS
    source_on_target_[level].CopyFrom(target_[level].intensity());
#endif
    server_->transform_source_to_target_.FromEigen(
        transform_source_to_target_);

    ICRGBDOdometryCudaKernelCaller<N>::ApplyICRGBDOdometryKernelCaller(
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

    PrintDebug("> Level %d, iter %d: loss = %f, avg loss = %f, inliers = %.0f\n",
               level, iter, loss, loss / inliers, inliers);

    bool is_success;
    Eigen::Matrix4d extrinsic;
    std::tie(is_success, extrinsic) =
        SolveJacobianSystemAndObtainExtrinsicMatrix(JtJ, Jtr);

    return std::make_tuple(is_success, extrinsic, loss);
}

template<size_t N>
std::tuple<bool, Eigen::Matrix4d, std::vector<std::vector<float>> >
ICRGBDOdometryCuda<N>::ComputeMultiScale() {
    bool is_success;
    Eigen::Matrix4d delta;
    float loss;

    std::vector<std::vector<float>> losses;
    for (int level = (int) (N - 1); level >= 0; --level) {
        std::vector<float> losses_on_level;

        for (int iter = 0;
             iter < option_.iteration_number_per_pyramid_level_[N - 1 - level];
             ++iter) {

            std::tie(is_success, delta, loss) =
                DoSingleIteration((size_t) level, iter);

            // TODO: check which is correct
            transform_source_to_target_ = delta.inverse()
                * transform_source_to_target_;
            losses_on_level.emplace_back(loss);

            if (!is_success) {
                PrintWarning("[ComputeOdometry] no solution!\n");
                return std::make_tuple(
                    false, Eigen::Matrix4d::Identity(),
                    losses);
            }
        }

        losses.emplace_back(losses_on_level);
    }

    return std::make_tuple(true, transform_source_to_target_, losses);
}
}
