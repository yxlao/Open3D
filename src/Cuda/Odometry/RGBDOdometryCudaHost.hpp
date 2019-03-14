//
// Created by wei on 11/9/18.
//

#pragma once

#include "RGBDOdometryCuda.h"
#include <Core/Core.h>

namespace open3d {
namespace cuda {
/**
 * Client end
 * TODO: Think about how do we use device_ ... we don't want copy
 * constructors for such a large system...
 */
template<size_t N>
RGBDOdometryCuda<N>::RGBDOdometryCuda() : device_(nullptr) {}

template<size_t N>
RGBDOdometryCuda<N>::~RGBDOdometryCuda() {
    Release();
}

template<size_t N>
void RGBDOdometryCuda<N>::SetParameters(
    const OdometryOption &option, float sigma) {
    assert(option_.iteration_number_per_pyramid_level_.size() == N);
    option_ = option;
    sigma_ = sigma;
}

template<size_t N>
void RGBDOdometryCuda<N>::SetIntrinsics(PinholeCameraIntrinsic intrinsics) {
    intrinsics_ = intrinsics;
}

template<size_t N>
bool RGBDOdometryCuda<N>::Create(int width, int height) {
    assert(width > 0 && height > 0);

    if (device_ != nullptr) {
        if (source_[0].width_ != width || source_[0].height_ != height) {
            PrintError("[RGBDOdometryCuda] Incompatible image size, "
                       "width: %d vs %d, height: %d vs %d, "
                       "@Create aborted.\n",
                       source_[0].width_, width, source_[0].height_, height);
            return false;
        }
        return true;
    }

    device_ = std::make_shared<RGBDOdometryCudaDevice<N>>();

    source_on_target_.Create(width, height);

    source_preprocessed_.Create(width, height);
    target_preprocessed_.Create(width, height);

    source_.Create(width, height);
    target_.Create(width, height);
    target_dx_.Create(width, height);
    target_dy_.Create(width, height);

    results_.Create(29); // 21 + 6 + 2
    correspondences_.Create(width * height);

    transform_source_to_target_ = Eigen::Matrix4d::Identity();

    UpdateDevice();
    return true;
}

template<size_t N>
void RGBDOdometryCuda<N>::Release() {
    source_preprocessed_.Release();
    target_preprocessed_.Release();

    source_.Release();
    target_.Release();
    target_dx_.Release();
    target_dy_.Release();

    source_on_target_.Release();

    results_.Release();
    correspondences_.Release();

    device_ = nullptr;
}

template<size_t N>
void RGBDOdometryCuda<N>::UpdateSigma(float sigma) {
    device_->sigma_ = sigma;
    device_->sqrt_coeff_D_ = sqrtf(sigma);
    device_->sqrt_coeff_I_ = sqrtf(1 - sigma);
}

template<size_t N>
void RGBDOdometryCuda<N>::UpdateDevice() {
    if (device_ != nullptr) {
        source_on_target_.UpdateDevice();
        device_->source_on_target_ = *source_on_target_.device_;

        source_preprocessed_.UpdateDevice();
        device_->source_input_ = *source_preprocessed_.device_;

        target_preprocessed_.UpdateDevice();
        device_->target_input_ = *target_preprocessed_.device_;

        source_.UpdateDevice();
        device_->source_ = *source_.device_;

        target_.UpdateDevice();
        device_->target_ = *target_.device_;

        target_dx_.UpdateDevice();
        device_->target_dx_ = *target_dx_.device_;

        target_dy_.UpdateDevice();
        device_->target_dy_ = *target_dy_.device_;

        device_->results_ = *results_.device_;
        device_->correspondences_ = *correspondences_.device_;

        /** Update parameters **/
        device_->min_depth_ = (float) option_.min_depth_;
        device_->max_depth_ = (float) option_.max_depth_;
        device_->max_depth_diff_ = (float) option_.max_depth_diff_;

        UpdateSigma(sigma_);

        device_->intrinsics_[0] = PinholeCameraIntrinsicCuda(intrinsics_);
        for (size_t i = 1; i < N; ++i) {
            device_->intrinsics_[i] = device_->intrinsics_[i - 1].Downsample();
        }
        device_->transform_source_to_target_.FromEigen(
            transform_source_to_target_);
    }
}

template<size_t N>
void RGBDOdometryCuda<N>::ExtractResults(std::vector<float> &results,
                                         Eigen::Matrix6d &JtJ,
                                         Eigen::Vector6d &Jtr,
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
void RGBDOdometryCuda<N>::Initialize(
    RGBDImageCuda &source, RGBDImageCuda &target) {
    assert(source.width_ == target.width_);
    assert(source.height_ == target.height_);

    bool success = Create(source.width_, source.height_);
    if (!success) {
        PrintError("[RGBDOdometryCuda] create failed, "
                   "@PrepareData aborted.\n");
        return;
    }

    /** Preprocess: truncate depth to nan values, then perform Gaussian **/
    source_preprocessed_.CopyFrom(source);
    target_preprocessed_.CopyFrom(target);
    RGBDOdometryCudaKernelCaller<N>::PreprocessDepth(*this);
    source_preprocessed_.depth_.Gaussian(source_[0].depth_,
        Gaussian3x3, false);
    target_preprocessed_.depth_.Gaussian(target_[0].depth_,
        Gaussian3x3, false);

    /** Preprocess: Gaussian intensities **/
    source_preprocessed_.intensity_.Gaussian(source_[0].intensity_,
        Gaussian3x3, false);
    target_preprocessed_.intensity_.Gaussian(target_[0].intensity_,
        Gaussian3x3, false);

    /** For debug **/
    source_preprocessed_.color_raw_.Gaussian(source_[0].color_raw_,
        Gaussian3x3, false);
    target_preprocessed_.color_raw_.Gaussian(target_[0].color_raw_,
        Gaussian3x3, false);
    auto im_source = source_[0].intensity_.DownloadImage();
    auto im_target = target_[0].intensity_.DownloadImage();

    /** Preprocess: normalize intensity
      * between pair (source_[0], target_[0]) **/
    device_->transform_source_to_target_.FromEigen(transform_source_to_target_);
    correspondences_.set_iterator(0);
    RGBDOdometryCudaKernelCaller<N>::NormalizeIntensity(*this);

    source_.Build(source_[0]);
    target_.Build(target_[0]);

    for (int i = 0; i < N; ++i) {
        /* Compute gradients */
        target_[i].depth_.Sobel(
            target_dx_[i].depth_, target_dy_[i].depth_, false);
        target_[i].intensity_.Sobel(
            target_dx_[i].intensity_, target_dy_[i].intensity_, false);
    }

    UpdateDevice();
}

template<size_t N>
std::tuple<bool, Eigen::Matrix4d, float>
RGBDOdometryCuda<N>::DoSingleIteration(size_t level, int iter) {
    results_.Memset(0);
    correspondences_.set_iterator(0);

#ifdef VISUALIZE_ODOMETRY_INLIERS
    source_on_target_[level].CopyFrom(target_[level].intensity());
#endif
    device_->transform_source_to_target_.FromEigen(
        transform_source_to_target_);

    Timer timer;
    timer.Start();
    RGBDOdometryCudaKernelCaller<N>::DoSingleIteration(*this, level);
    timer.Stop();

#ifdef VISUALIZE_ODOMETRY_INLIERS
    cv::Mat im = source_on_target_[level].DownloadMat();
        cv::imshow("source_on_target", im);
        cv::waitKey(-1);
#endif

    std::vector<float> results = results_.DownloadAll();

    Eigen::Matrix6d JtJ;
    Eigen::Vector6d Jtr;
    float loss, inliers;
    ExtractResults(results, JtJ, Jtr, loss, inliers);
    PrintDebug("> Level %d, iter %d: loss = %f, avg loss = %f, inliers = %.0f\n",
               level, iter, loss, loss / inliers, inliers);

    bool is_success;
    Eigen::Matrix4d extrinsic;
    std::tie(is_success, extrinsic) =
        SolveJacobianSystemAndObtainExtrinsicMatrix(JtJ, Jtr);

    return std::make_tuple(is_success, extrinsic, loss / inliers);
}

template<size_t N>
std::tuple<bool, Eigen::Matrix4d, std::vector<std::vector<float>>>
RGBDOdometryCuda<N>::ComputeMultiScale() {
    bool is_success;
    Eigen::Matrix4d delta;
    float loss;

    std::vector<std::vector<float>> losses;
    for (int level = (int) (N - 1); level >= 0; --level) {
        std::vector<float> losses_on_level;

//        float factor = std::pow(1.39f, (N - 1 - level));
//        UpdateSigma(std::min(sigma_ * factor, 0.968f));

        for (int iter = 0;
             iter < option_.iteration_number_per_pyramid_level_[N - 1 - level];
             ++iter) {

            std::tie(is_success, delta, loss) =
                DoSingleIteration((size_t) level, iter);
            transform_source_to_target_ = delta * transform_source_to_target_;
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

template<size_t N>
Eigen::Matrix6d RGBDOdometryCuda<N>::ComputeInformationMatrix() {
    results_.Memset(0);

    RGBDOdometryCudaKernelCaller<N>::ComputeInformationMatrix(*this);
    std::vector<float> results = results_.DownloadAll();

    Eigen::Matrix6d JtJ;
    Eigen::Vector6d Jtr; // dummy
    float loss, inliers; // dummy
    ExtractResults(results, JtJ, Jtr, loss, inliers);

    return JtJ;
}
} // cuda
} // open3d