//
// Created by wei on 11/9/18.
//

#pragma once

#include "RGBDOdometryCuda.h"

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
    const odometry::OdometryOption &option, float sigma) {
    assert(option_.iteration_number_per_pyramid_level_.size() == N);
    option_ = option;
    sigma_ = sigma;
}

template<size_t N>
void RGBDOdometryCuda<N>::SetIntrinsics(camera::PinholeCameraIntrinsic
intrinsics) {
    intrinsics_ = intrinsics;
}

template<size_t N>
bool RGBDOdometryCuda<N>::Create(int width, int height) {
    assert(width > 0 && height > 0);

    if (device_ != nullptr) {
        if (source_depth_[0].width_ != width
        || source_depth_[0].height_ != height) {
            utility::PrintError("[RGBDOdometryCuda] Incompatible image size, "
                       "width: %d vs %d, height: %d vs %d, "
                       "@Create aborted.\n",
                       source_depth_[0].width_, width,
                       source_depth_[0].height_, height);
            return false;
        }
        return true;
    }

    device_ = std::make_shared<RGBDOdometryCudaDevice<N>>();

    source_input_.Create(width, height);
    target_input_.Create(width, height);

    for (int i = 0; i < N; ++i) {
        source_depth_[i].Create(width >> i, height >> i);
        source_intensity_[i].Create(width >> i, height >> i);
        target_depth_[i].Create(width >> i , height >> i);
        target_intensity_[i].Create(width >> i, height >> i);

        target_depth_dx_[i].Create(width >> i, height >> i);
        target_depth_dy_[i].Create(width >> i, height >> i);
        target_intensity_dx_[i].Create(width >> i, height >> i);
        target_intensity_dy_[i].Create(width >> i, height >> i);
    }

    results_.Create(29); // 21 + 6 + 2
    correspondences_.Create(width * height);

    transform_source_to_target_ = Eigen::Matrix4d::Identity();

    UpdateDevice();
    return true;
}

template<size_t N>
void RGBDOdometryCuda<N>::Release() {
    source_input_.Release();
    target_input_.Release();

    for (int i = 0; i < N; ++i) {
        source_depth_[i].Release();
        source_intensity_[i].Release();
        target_depth_[i].Release();
        target_intensity_[i].Release();

        target_depth_dx_[i].Release();
        target_depth_dy_[i].Release();
        target_intensity_dx_[i].Release();
        target_intensity_dy_[i].Release();
    }

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
        source_input_.UpdateDevice();
        device_->source_input_ = *source_input_.device_;

        target_input_.UpdateDevice();
        device_->target_input_ = *target_input_.device_;

        for (int i = 0; i < N; ++i) {
            source_depth_[i].UpdateDevice();
            device_->source_depth_[i] = *source_depth_[i].device_;
            source_intensity_[i].UpdateDevice();
            device_->source_intensity_[i] = *source_intensity_[i].device_;

            target_depth_[i].UpdateDevice();
            device_->target_depth_[i] = *target_depth_[i].device_;
            target_intensity_[i].UpdateDevice();
            device_->target_intensity_[i] = *target_intensity_[i].device_;

            target_depth_dx_[i].UpdateDevice();
            device_->target_depth_dx_[i] = *target_depth_dx_[i].device_;
            target_depth_dy_[i].UpdateDevice();
            device_->target_depth_dy_[i] = *target_depth_dy_[i].device_;

            target_intensity_dx_[i].UpdateDevice();
            device_->target_intensity_dx_[i] = *target_intensity_dx_[i].device_;
            target_intensity_dy_[i].UpdateDevice();
            device_->target_intensity_dy_[i] = *target_intensity_dy_[i].device_;
        }

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
        utility::PrintError("[RGBDOdometryCuda] create failed, "
                   "@PrepareData aborted.\n");
        return;
    }

    source_input_.CopyFrom(source);
    target_input_.CopyFrom(target);

    /** Preprocess: truncate depth to nan values, then perform Gaussian **/
    ImageCudaf source_depth_preprocessed, source_intensity_preprocessed;
    ImageCudaf target_depth_preprocessed, target_intensity_preprocessed;
    source_depth_preprocessed.Create(source.width_, source.height_);
    source_intensity_preprocessed.Create(source.width_, source.height_);
    target_depth_preprocessed.Create(source.width_, source.height_);
    target_intensity_preprocessed.Create(source.width_, source.height_);
    RGBDOdometryCudaKernelCaller<N>::PreprocessInput(*this,
        source_depth_preprocessed, source_intensity_preprocessed,
        target_depth_preprocessed, target_intensity_preprocessed);

    /** Preprocess: Smooth **/
    source_depth_preprocessed.Gaussian(source_depth_[0], Gaussian3x3);
    source_intensity_preprocessed.Gaussian(source_intensity_[0], Gaussian3x3);
    target_depth_preprocessed.Gaussian(target_depth_[0], Gaussian3x3);
    target_intensity_preprocessed.Gaussian(target_intensity_[0], Gaussian3x3);

    /** Preprocess: normalize intensity between pair (source_[0], target_[0]) **/
    device_->transform_source_to_target_.FromEigen(transform_source_to_target_);
    correspondences_.set_iterator(0);
    RGBDOdometryCudaKernelCaller<N>::NormalizeIntensity(*this);

    /* Downsample */
    for (int i = 1; i < N; ++i) {
        source_depth_[i - 1].Downsample(source_depth_[i], BoxFilter);
        target_depth_[i - 1].Downsample(target_depth_[i], BoxFilter);

        auto tmp = source_intensity_[i - 1].Gaussian(Gaussian3x3);
        tmp.Downsample(source_intensity_[i], BoxFilter);
        tmp = target_intensity_[i - 1].Gaussian(Gaussian3x3);
        tmp.Downsample(target_intensity_[i], BoxFilter);
    }

    /* Compute gradients */
    for (int i = 0; i < N; ++i) {
        target_depth_[i].Sobel(target_depth_dx_[i], target_depth_dy_[i]);
        target_intensity_[i].Sobel(target_intensity_dx_[i], target_intensity_dy_[i]);
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

    utility::Timer timer;
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
    utility::PrintDebug("> Level %d, iter %d: loss = %f, avg loss = %f, "
                        "inliers = %"
                ".0f\n",
               level, iter, loss, loss / inliers, inliers);

    bool is_success;
    Eigen::Matrix4d extrinsic;
    std::tie(is_success, extrinsic) =
        utility::SolveJacobianSystemAndObtainExtrinsicMatrix(JtJ, Jtr);

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

        for (int iter = 0;
             iter < option_.iteration_number_per_pyramid_level_[N - 1 - level];
             ++iter) {

            std::tie(is_success, delta, loss) =
                DoSingleIteration((size_t) level, iter);
            transform_source_to_target_ = delta * transform_source_to_target_;
            losses_on_level.emplace_back(loss);

            if (!is_success) {
                utility::PrintWarning("[ComputeOdometry] no solution!\n");
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