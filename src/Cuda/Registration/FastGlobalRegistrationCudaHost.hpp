//
// Created by wei on 1/21/19.
//

#include <Cuda/Geometry/NNCuda.h>
#include "FastGlobalRegistrationCuda.h"

namespace open3d {
namespace cuda {
void FastGlobalRegistrationCuda::Create() {
    if (server_ == nullptr) {
        server_ = std::make_shared<FastGlobalRegistrationCudaDevice>();
        results_.Create(28);
    }
}

void FastGlobalRegistrationCuda::Release() {
    if (server_ == nullptr && server_.use_count() == 1) {
        results_.Release();
        source_.Release();
        target_.Release();
        corres_source_to_target_.Release();
        corres_target_to_source_.Release();
        corres_mutual_.Release();
        corres_final_.Release();
    }
    server_ = nullptr;
}

void FastGlobalRegistrationCuda::UpdateServer() {
    if (server_ == nullptr) {
        PrintError("Server not initialized!\n");
        return;
    }

    server_->results_ = *results_.server();
    server_->source_ = *source_.server();
    server_->target_ = *target_.server();
    server_->corres_source_to_target_ = *corres_source_to_target_.server_;
    server_->corres_target_to_source_ = *corres_target_to_source_.server_;
}

void FastGlobalRegistrationCuda::ExtractResults(
    Eigen::Matrix6d &JtJ, Eigen::Vector6d &Jtr, float &rmse) {
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
    rmse = downloaded_result[cnt];
}

void FastGlobalRegistrationCuda::Initialize(PointCloud &source,
                                            PointCloud &target) {

    source_.Create(VertexWithNormal, (int)source.points_.size());
    source_.Upload(source);
    target_.Create(VertexWithNormal, (int)target.points_.size());
    target_.Upload(target);

    /** Extract feature from original point clouds **/
    source_feature_extractor_.Compute(
        source, KDTreeSearchParamHybrid(0.25, 100));
    target_feature_extractor_.Compute(
        target, KDTreeSearchParamHybrid(0.25, 100));
    source_features_ = source_feature_extractor_.fpfh_features_;
    target_features_ = target_feature_extractor_.fpfh_features_;

    /* 1) Initial Matching */
    nn_source_to_target_.NNSearch(source_features_, target_features_);
    corres_source_to_target_.SetCorrespondenceMatrix(
        nn_source_to_target_.nn_idx_);
    corres_source_to_target_.Compress();

    nn_target_to_source_.NNSearch(target_features_, source_features_);
    corres_target_to_source_.SetCorrespondenceMatrix(
        nn_target_to_source_.nn_idx_);
    corres_target_to_source_.Compress();
    UpdateServer();

    PrintInfo("(%d x %d, %d) - (%d x %d, %d)\n",
              corres_source_to_target_.matrix_.max_rows_,
              corres_source_to_target_.matrix_.max_cols_,
              corres_source_to_target_.indices_.size(),
              corres_target_to_source_.matrix_.max_rows_,
              corres_target_to_source_.matrix_.max_cols_,
              corres_target_to_source_.indices_.size());

    /* 2) Reciprocity Test */
    corres_mutual_.Create(source.points_.size());
    server_->corres_mutual_ = *corres_mutual_.server();
    FastGlobalRegistrationCudaKernelCaller::ReciprocityTest(*this);

    corres_final_.Create(corres_mutual_.size());
    server_->corres_final_ = *corres_final_.server();
    FastGlobalRegistrationCudaKernelCaller::TupleTest(*this);

    l_.Create(corres_final_.size());
    l_.Fill(1.0f);
    server_->l_ = *l_.server();

    double scale_global = NormalizePointClouds();
    server_->scale_global_ = (float) scale_global;
    server_->par_ = (float) scale_global;

    transform_normalized_source_to_target_ = Eigen::Matrix4d::Identity();
}

double FastGlobalRegistrationCuda::NormalizePointClouds() {
    mean_source_ =
        FastGlobalRegistrationCudaKernelCaller::ComputePointCloudSum(
            source_);
    mean_source_ /= source_.points().size();
    double scale_source =
        FastGlobalRegistrationCudaKernelCaller::NormalizePointCloud(
            source_, mean_source_);

    mean_target_ =
        FastGlobalRegistrationCudaKernelCaller::ComputePointCloudSum(
            target_);
    mean_target_ /= target_.points().size();
    double scale_target =
        FastGlobalRegistrationCudaKernelCaller::NormalizePointCloud(
            target_, mean_target_);

    double scale_global = std::max(scale_source, scale_target);
    FastGlobalRegistrationCudaKernelCaller::RescalePointCloud(
        source_, scale_global);
    FastGlobalRegistrationCudaKernelCaller::RescalePointCloud(
        target_, scale_global);

    return scale_global;
}

namespace {
Eigen::Matrix4d GetTransformationOriginalScale(
    const Eigen::Matrix4d &transformation,
    const Eigen::Vector3d &mean_source,
    const Eigen::Vector3d &mean_target,
    const double scale_global) {
    Eigen::Matrix3d R = transformation.block<3, 3>(0, 0);
    Eigen::Vector3d t = transformation.block<3, 1>(0, 3);
    Eigen::Matrix4d transtemp = Eigen::Matrix4d::Zero();
    transtemp.block<3, 3>(0, 0) = R;
    transtemp.block<3, 1>(0, 3) =
        -R * mean_source + t * scale_global + mean_target;
    transtemp(3, 3) = 1;
    return transtemp;
}
}

RegistrationResultCuda FastGlobalRegistrationCuda::DoSingleIteration(int iter) {
    RegistrationResultCuda result;

    results_.Memset(0);
    FastGlobalRegistrationCudaKernelCaller::
    ComputeResultsAndTransformation(*this);

    Eigen::Matrix6d JtJ;
    Eigen::Vector6d Jtr;
    float rmse;
    ExtractResults(JtJ, Jtr, rmse);
    std::cout << "gpu JtJ: " << JtJ << std::endl;
    std::cout << "gpu Jtr: " << Jtr.transpose() << std::endl;
    std::cout << "gpu rmse: " << rmse << std::endl;

    bool success;
    Eigen::VectorXd xi;
    std::tie(success, xi) = SolveLinearSystem(-JtJ, Jtr);
    Eigen::Matrix4d delta = TransformVector6dToMatrix4d(xi);
    transform_normalized_source_to_target_ =
        delta * transform_normalized_source_to_target_;
    source_.Transform(delta);
    std::cout << "gpu delta: " << delta << std::endl;
    std::cout << "gpu trans: " << transform_normalized_source_to_target_ <<
    std::endl;

    result.transformation_ = GetTransformationOriginalScale(
        transform_normalized_source_to_target_, mean_source_, mean_target_,
        server_->scale_global_);
    result.inlier_rmse_ = rmse;
    PrintInfo("iter: %d, rmse: %f\n", iter, rmse);

    if (iter % 4 == 0 && server_->par_ > 0.0f) {
        server_->par_ /= 1.4f;
        std::cout << "gpu par: " << server_->par_ << std::endl;
    }

    return result;
};
}
}
