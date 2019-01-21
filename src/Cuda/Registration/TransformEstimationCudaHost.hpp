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

    correspondences_.Create(source.points_.size(), 1);

    /** CPU part **/
    max_correspondence_distance_ = max_correspondence_distance;
    source_cpu_ = source;
    target_cpu_ = target;
    kdtree_.SetGeometry(target_cpu_);
    corres_matrix_ = Eigen::Matrix<int, -1, -1, Eigen::RowMajor>(
        source_cpu_.points_.size(), 1);

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
            corres_matrix_(i, 0) = found ? indices[0] : -1;
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

        server_->correspondences_ = *correspondences_.server();

        server_->results_ = *results_.server();
    }
}

RegistrationResultCuda TransformEstimationPointToPointCuda::
ComputeResultsAndTransformation() {
    RegistrationResultCuda result;

    Eigen::Matrix3d Sigma;
    Eigen::Vector3d mean_source, mean_target;
    float sigma_x2, rmse;

    results_.Memset(0);
    TransformEstimationPointToPointCudaKernelCaller::
    ComputeMeansKernelCaller(*this);

    UnpackResults(Sigma, mean_source, mean_target, sigma_x2, rmse);
    int inliers = correspondences_.indices_.size();
    mean_source /= inliers;
    mean_target /= inliers;
    server_->source_mean_.FromEigen(mean_source);
    server_->target_mean_.FromEigen(mean_target);

    Eigen::MatrixXd source_mat(3, inliers);
    Eigen::MatrixXd target_mat(3, inliers);

    int cnt = 0;
    for (size_t i = 0; i < corres_matrix_.rows(); i++) {
        if (corres_matrix_(i, 0) == -1) continue;
        source_mat.block<3, 1>(0, cnt)
            = source_cpu_.points_[i];
        target_mat.block<3, 1>(0, cnt)
            = target_cpu_.points_[corres_matrix_(i, 0)];
        ++cnt;
    }
    assert(cnt == inliers);
//    std::cout << mean_source.transpose()
//              << " vs "
//              << source_mat.rowwise().mean().transpose()
//              << std::endl;
//    std::cout << mean_target.transpose()
//              << " vs "
//              << target_mat.rowwise().mean().transpose()
//              << std::endl;

    Eigen::MatrixXd X = source_mat.colwise() - source_mat.rowwise().mean();
    Eigen::MatrixXd Y = target_mat.colwise() - target_mat.rowwise().mean();

//    std::cout << X << std::endl;

    Eigen::Matrix3d XYt = Y * X.transpose() / inliers;

    TransformEstimationPointToPointCudaKernelCaller::
    ComputeResultsAndTransformationKernelCaller(*this);

    UnpackResults(Sigma, mean_source, mean_target, sigma_x2, rmse);
    Sigma /= inliers;
    mean_source /= inliers;
    mean_target /= inliers;
    sigma_x2 /= inliers;

//    std::cout << XYt << std::endl;
//    std::cout << Sigma << std::endl;

    Eigen::JacobiSVD<Eigen::Matrix3d> svd(Sigma,
        Eigen::ComputeFullU | Eigen::ComputeFullV);
    Eigen::Matrix3d U = svd.matrixU(), V = svd.matrixV();
    Eigen::Vector3d d = svd.singularValues();
    Eigen::Matrix3d S = Eigen::Matrix3d::Identity();
    S(2, 2) = Sigma.determinant() >= 0 ? 1 : -1;

    Eigen::Matrix3d R = U * S * (V.transpose());
    double scale = with_scaling_ ?
                   (1.0 / sigma_x2) * (S * d).sum()
                   : 1.0;
    Eigen::Vector3d t = mean_target - scale * R * mean_source;

    Eigen::Matrix4d extrinsic = Eigen::Matrix4d::Identity();
    extrinsic.block<3, 3>(0, 0) = scale * R;
    extrinsic.block<3, 1>(0, 3) = t;

    result.fitness_ = float(inliers) / source_.points().size();
    result.inlier_rmse_ = sqrt(rmse / inliers);
    result.transformation_ = extrinsic;
    // result.correspondences_ = correspondences_;

    return result;
}

void TransformEstimationPointToPointCuda::UnpackResults(
    Eigen::Matrix3d &Sigma,
    Eigen::Vector3d &mean_source,
    Eigen::Vector3d &mean_target,
    float &sigma_source2, float &rmse) {

    std::vector<float> downloaded_result = results_.DownloadAll();
    int cnt = 0;
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            Sigma(i, j) = downloaded_result[cnt];
            ++cnt;
        }
    }
    for (int i = 0; i < 3; ++i) {
        mean_source(i) = downloaded_result[cnt];
        ++cnt;
    }
    for (int i = 0; i < 3; ++i) {
        mean_target(i) = downloaded_result[cnt];
        ++cnt;
    }
    sigma_source2 = downloaded_result[cnt]; ++cnt;
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

        server_->correspondences_ = *correspondences_.server();

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