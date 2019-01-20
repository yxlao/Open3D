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
        for (int i = 0; i < (int)source_cpu_.points_.size(); ++i) {
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
    result.fitness_ = float(inliers) / target_.points().size();
    result.inlier_rmse_ = rmse / inliers;
    result.transformation_ = extrinsic;
    // result.correspondences_ = correspondences_;

    return result;
}
}
}