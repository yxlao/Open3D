//
// Created by wei on 4/4/19.
//

#include "ScalableVolumeRegistrationCuda.h"


namespace open3d {
namespace cuda {

ScalableVolumeRegistrationCuda::ScalableVolumeRegistrationCuda() {
    Create();
}

ScalableVolumeRegistrationCuda::~ScalableVolumeRegistrationCuda() {
    Release();
}

void ScalableVolumeRegistrationCuda::Create() {
    if (device_ != nullptr) {
        utility::PrintWarning("[ScalbleVolumeRegistration] Already created, "
                              "abort\n");
        return ;
    }

    device_ = std::make_shared<ScalableVolumeRegistrationCudaDevice>();
    results_.Create(29);
}

void ScalableVolumeRegistrationCuda::Release(){
    device_ = nullptr;

    source_.Release();
    target_.Release();
    source_property_.Release();
    results_.Release();
}

void ScalableVolumeRegistrationCuda::UpdateDevice(){
    device_->source_ = *source_.device_;
    device_->target_ = *target_.device_;
    device_->source_property_ = *source_property_.device_;
    device_->results_ = *results_.device_;

    device_->trans_source_to_target_.FromEigen(trans_source_to_target_);
}

void ScalableVolumeRegistrationCuda::Initialize(ScalableTSDFVolumeCuda &source,
                                                ScalableTSDFVolumeCuda &target,
                                                const Eigen::Matrix4d &init) {
    source_ = source;
    target_ = target;

    source_active_subvolumes_ = source_.active_subvolume_entry_array_.size();
    source_property_ = ScalableTSDFVolumeProcessorCuda(
        source_.N_, source_active_subvolumes_);
    source_property_.ComputeGradient(source_);

    trans_source_to_target_ = init;

    UpdateDevice();
}

RegistrationResultCuda ScalableVolumeRegistrationCuda::DoSingleIteration(
    int iter) {

    RegistrationResultCuda delta;
    delta.transformation_ = Eigen::Matrix4d::Identity();
    delta.fitness_ = delta.inlier_rmse_ = 0;

    delta = BuildAndSolveLinearSystem();

    utility::PrintDebug("Iteration %d: inlier rmse = %f, inliers = %f\n",
                        iter, delta.inlier_rmse_, delta.fitness_);

    trans_source_to_target_ =
//        delta.transformation_.inverse() * trans_source_to_target_;
        trans_source_to_target_ * delta.transformation_.inverse() ;
    device_->trans_source_to_target_.FromEigen(trans_source_to_target_);

    return delta;
}

RegistrationResultCuda ScalableVolumeRegistrationCuda::BuildAndSolveLinearSystem() {
    RegistrationResultCuda result;

    results_.Memset(0);
    ScalableVolumeRegistrationCudaKernelCaller::BuildLinearSystem(*this);

    Eigen::Matrix6d JtJ;
    Eigen::Vector6d Jtr;
    float rmse, inliers;
    ExtractResults(JtJ, Jtr, rmse, inliers);
//    std::cout << JtJ << "\n" << Jtr.transpose() << "\n";
    bool is_success;
    Eigen::Matrix4d extrinsic;
    std::tie(is_success, extrinsic) =
        utility::SolveJacobianSystemAndObtainExtrinsicMatrix(JtJ, Jtr);

    result.fitness_ = inliers;
    result.inlier_rmse_ = sqrtf(rmse / inliers);
    result.transformation_ = extrinsic;

    return result;
}

void ScalableVolumeRegistrationCuda::ExtractResults(
    Eigen::Matrix6d &JtJ, Eigen::Vector6d &Jtr, float &rmse, float &inliers) {
    std::vector<float> downloaded_result = results_.DownloadAll();
    int cnt = 0;
    for (int i = 0; i < 6; ++i) {
        for (int j = i; j < 6; ++j) {
            JtJ(i, j) = JtJ(j, i) = downloaded_result[cnt];
            ++cnt;
        }
    }
    for (int i = 0; i < 6; ++i) {
        Jtr(i) = downloaded_result[cnt];
        ++cnt;
    }
    rmse = downloaded_result[cnt]; ++cnt;
    inliers = downloaded_result[cnt];
}
}
}