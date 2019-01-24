//
// Created by wei on 1/11/19.
//

#pragma once

#include "TransformEstimationCuda.h"

namespace open3d {
namespace cuda {

void TransformEstimationCuda::Initialize(PointCloud &source,
                                         PointCloud &target,
                                         float max_correspondence_distance) {
    /** GPU part **/
    source_.Create(VertexWithNormalAndColor, source.points_.size());
    source_.Upload(source);

    target_.Create(VertexWithNormalAndColor, target.points_.size());
    target_.Upload(target);

    correspondences_.Create(1, source.points_.size());

    /** CPU part **/
    max_correspondence_distance_ = max_correspondence_distance;
    source_cpu_ = source;
    target_cpu_ = target;
    kdtree_.SetGeometry(target_cpu_);
    corres_matrix_ = Eigen::MatrixXi(1, source.points_.size());

    UpdateServer();
}

void TransformEstimationCuda::GetCorrespondences() {
#ifdef _OPENMP
#pragma omp parallel
    {
#endif
#ifdef _OPENMP
#pragma omp for nowait
#endif
        for (int i = 0; i < (int) source_cpu_.points_.size(); ++i) {
            std::vector<int> indices(1);
            std::vector<double> dists(1);

            bool found = kdtree_.SearchHybrid(source_cpu_.points_[i],
                                              max_correspondence_distance_, 1,
                                              indices, dists) > 0;
            corres_matrix_(0, i) = found ? indices[0] : -1;
        }
#ifdef _OPENMP
    }
#endif
    correspondences_.SetCorrespondenceMatrix(corres_matrix_);
    correspondences_.Compress();

    UpdateServer();
}

void TransformEstimationCuda::TransformSourcePointCloud(
    const Eigen::Matrix4d &source_to_target) {
    source_.Transform(source_to_target);
    source_cpu_.Transform(source_to_target);
}

void TransformEstimationCuda::ExtractResults(
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

/** TransformEstimationPointToPointCuda **/
void TransformEstimationPointToPointCuda::Create() {
    server_ = std::make_shared<TransformEstimationPointToPointCudaDevice>();

    /** 9 + 3 + 3 + 1 + 1 **/
    results_.Create(17);
}

void TransformEstimationPointToPointCuda::Release() {
    if (server_ != nullptr && server_.use_count() == 1) {
        source_.Release();
        target_.Release();
        correspondences_.Release();

        results_.Release();
    }
    server_ = nullptr;
}

void TransformEstimationPointToPointCuda::UpdateServer() {
    if (server_ != nullptr) {
        server_->source_ = *source_.server();
        server_->target_ = *target_.server();

        server_->correspondences_ = *correspondences_.server_;

        server_->results_ = *results_.server();
    }
}

RegistrationResultCuda TransformEstimationPointToPointCuda::
ComputeResultsAndTransformation() {
    RegistrationResultCuda result;
    results_.Memset(0);
    int inliers = correspondences_.indices_.size();

    /** Pass 1: sum reduction means **/
    TransformEstimationPointToPointCudaKernelCaller::
    ComputeSumsKernelCaller(*this);

    Eigen::Vector3d source_mean, target_mean;
    UnpackSums(source_mean, target_mean);
    source_mean /= inliers;
    target_mean /= inliers;
    server_->source_mean_.FromEigen(source_mean);
    server_->target_mean_.FromEigen(target_mean);

    /** Pass 2: sum reduction Sigma **/
    TransformEstimationPointToPointCudaKernelCaller::
    ComputeResultsAndTransformationKernelCaller(*this);

    Eigen::Matrix3d Sigma;
    float source_sigma2, rmse;
    UnpackSigmasAndRmse(Sigma, source_sigma2, rmse);
    Sigma /= inliers;
    source_sigma2 /= inliers;

    /** Solve linear system **/
    Eigen::JacobiSVD<Eigen::Matrix3d> svd(Sigma,
        Eigen::ComputeFullU | Eigen::ComputeFullV);
    const Eigen::Matrix3d &U = svd.matrixU();
    const Eigen::Matrix3d &V = svd.matrixV();
    const Eigen::Vector3d &d = svd.singularValues();
    Eigen::Matrix3d S = Eigen::Matrix3d::Identity();
    S(2, 2) = Sigma.determinant() >= 0 ? 1 : -1;

    Eigen::Matrix3d R = U * S * (V.transpose());
    double scale = with_scaling_ ?
                   (1.0 / source_sigma2) * (S * d).sum()
                   : 1.0;
    Eigen::Vector3d t = target_mean - scale * R * source_mean;

    Eigen::Matrix4d extrinsic = Eigen::Matrix4d::Identity();
    extrinsic.block<3, 3>(0, 0) = scale * R;
    extrinsic.block<3, 1>(0, 3) = t;

    result.fitness_ = float(inliers) / source_.points().size();
    result.inlier_rmse_ = sqrt(rmse / inliers);
    result.transformation_ = extrinsic;
    // result.correspondences_ = correspondences_;

    return result;
}

void TransformEstimationPointToPointCuda::UnpackSums(
    Eigen::Vector3d &sum_source, Eigen::Vector3d &sum_target) {

    std::vector<float> downloaded_result = results_.DownloadAll();

    int cnt = 9;
    for (int i = 0; i < 3; ++i) {
        sum_source(i) = downloaded_result[cnt];
        ++cnt;
    }
    for (int i = 0; i < 3; ++i) {
        sum_target(i) = downloaded_result[cnt];
        ++cnt;
    }
}

void TransformEstimationPointToPointCuda::UnpackSigmasAndRmse(
    Eigen::Matrix3d &Sigma, float &source_sigma2, float &rmse) {

    std::vector<float> downloaded_result = results_.DownloadAll();

    int cnt = 0;
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            Sigma(i, j) = downloaded_result[cnt];
            ++cnt;
        }
    }

    cnt = 15;
    source_sigma2 = downloaded_result[cnt]; ++cnt;
    rmse = downloaded_result[cnt];
}

/** TransformEstimationPointToPlaneCuda **/
void TransformEstimationPointToPlaneCuda::Create() {
    server_ = std::make_shared<TransformEstimationPointToPlaneCudaDevice>();
    results_.Create(28);
}

void TransformEstimationPointToPlaneCuda::Release() {
    if (server_ != nullptr && server_.use_count() == 1) {
        source_.Release();
        target_.Release();
        correspondences_.Release();

        results_.Release();
    }
    server_ = nullptr;
}

void TransformEstimationPointToPlaneCuda::UpdateServer() {
    if (server_ != nullptr) {
        server_->source_ = *source_.server();
        server_->target_ = *target_.server();

        server_->correspondences_ = *correspondences_.server_;

        server_->results_ = *results_.server();
    }
}

RegistrationResultCuda TransformEstimationPointToPlaneCuda::
ComputeResultsAndTransformation() {
    RegistrationResultCuda result;

    results_.Memset(0);
    TransformEstimationPointToPlaneCudaKernelCaller::
    ComputeResultsAndTransformationKernelCaller(*this);

    Eigen::Matrix6d JtJ;
    Eigen::Vector6d Jtr;
    float rmse;
    ExtractResults(JtJ, Jtr, rmse);

    bool is_success;
    Eigen::Matrix4d extrinsic;
    std::tie(is_success, extrinsic) =
        SolveJacobianSystemAndObtainExtrinsicMatrix(JtJ, Jtr);

    int inliers = correspondences_.indices_.size();
    result.fitness_ = float(inliers) / source_.points().size();
    result.inlier_rmse_ = sqrt(rmse / inliers);
    result.transformation_ = extrinsic;
    // result.correspondences_ = correspondences_;

    return result;
}

} // cuda
} // open3d