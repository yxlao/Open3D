//
// Created by wei on 1/11/19.
//

#include "ColoredICPCuda.h"
#include <Core/Core.h>

namespace open3d {
namespace cuda {

TransformEstimationCudaForColoredICP::TransformEstimationCudaForColoredICP(
    float lambda_geometric) :
    lambda_geometric_(lambda_geometric) {
    if (lambda_geometric_ < 0 || lambda_geometric_ > 1.0)
        lambda_geometric_ = 0.968f;

    Create();
}

TransformEstimationCudaForColoredICP::~TransformEstimationCudaForColoredICP() {
    Release();
}

void TransformEstimationCudaForColoredICP::Create() {
    server_ = std::make_shared<TransformEstimationCudaForColoredICPDevice>();
    /* 21 JtJ + 6 Jtr + 1 RMSE */
    results_.Create(28);
}

void TransformEstimationCudaForColoredICP::Release() {
    if (server_ != nullptr && server_.use_count() == 1) {
        target_color_gradient_.Release();

        source_.Release();
        target_.Release();
        results_.Release();
    }
    server_ = nullptr;
}

void TransformEstimationCudaForColoredICP::UpdateServer() {
    if (server_ != nullptr) {
        server_->source_ = *source_.server();
        server_->target_ = *target_.server();
        server_->target_color_gradient_ = *target_color_gradient_.server();

        server_->correspondences_ = *correspondences_.server();

        server_->results_ = *results_.server();

        server_->lambda_geometric_ = lambda_geometric_;
        server_->sqrt_coeff_G_ = sqrtf(lambda_geometric_);
        server_->sqrt_coeff_I_ = sqrtf(1 - lambda_geometric_);
    }
}

void TransformEstimationCudaForColoredICP::Initialize(
    PointCloud &source, PointCloud &target, float max_correspondence_distance) {
    /** GPU part **/
    source_.Create(VertexWithNormalAndColor, source.points_.size());
    source_.Upload(source);

    target_.Create(VertexWithNormalAndColor, target.points_.size());
    target_.Upload(target);

    correspondences_.Create(source.points_.size(), 1);
    target_color_gradient_.Resize(target.points_.size());

    UpdateServer();

    /** CPU part **/
    max_correspondence_distance_ = max_correspondence_distance;
    source_cpu_ = source;
    target_cpu_ = target;
    kdtree_.SetGeometry(target_cpu_);

    /** Colored ICP specific **/
    ComputeColorGradients(target_cpu_, kdtree_,
                          KDTreeSearchParamHybrid(
                              max_correspondence_distance_ * 2, 30));
}

void TransformEstimationCudaForColoredICP::ComputeColorGradients(
    PointCloud &target,
    KDTreeFlann &kdtree, const KDTreeSearchParamHybrid &search_param) {

    /** Initialize correspondence matrix **/
    Eigen::Matrix<int, -1, -1, Eigen::RowMajor> corres_matrix
        = Eigen::Matrix<int, -1, -1, Eigen::RowMajor>::Constant(
            target.points_.size(), search_param.max_nn_, -1);

    /** OpenMP parallel K-NN search **/
#ifdef _OPENMP
    {
#endif
#ifdef _OPENMP
#pragma omp for nowait
#endif
        for (int i = 0; i < target.points_.size(); ++i) {
            std::vector<int> indices(search_param.max_nn_);
            std::vector<double> dists(search_param.max_nn_);

            if (kdtree.SearchHybrid(target.points_[i],
                                    search_param.radius_, search_param.max_nn_,
                                    indices, dists) >= 3) {
                corres_matrix.block(i, 0, 1, indices.size()) =
                    Eigen::Map<Eigen::RowVectorXi>(
                        indices.data(), indices.size());
            }
        }
#ifdef _OPENMP
    }
#endif

    /** Upload correspondences **/
    CorrespondenceSetCuda corres_for_color_gradient;
    corres_for_color_gradient.SetCorrespondenceMatrix(corres_matrix);
    corres_for_color_gradient.Compress();

    /** Run GPU color_gradient intialization **/
    TransformEstimationCudaForColoredICPKernelCaller::
    ComputeColorGradeintKernelCaller(*this, corres_for_color_gradient);
}

void TransformEstimationCudaForColoredICP::GetCorrespondences() {
    Eigen::Matrix<int, -1, -1, Eigen::RowMajor> corres_matrix
        = Eigen::Matrix<int, -1, -1, Eigen::RowMajor>::Constant(
            source_cpu_.points_.size(), 1, -1);
#ifdef _OPENMP
    {
#endif
#ifdef _OPENMP
#pragma omp for nowait
#endif
        for (int i = 0; i < source_cpu_.points_.size(); ++i) {
            std::vector<int> indices(1);
            std::vector<double> dists(1);

            if (kdtree_.SearchHybrid(source_cpu_.points_[i],
                                     max_correspondence_distance_, 1,
                                     indices, dists) > 0) {
                corres_matrix(i, 0) = indices[0];
            }
        }
#ifdef _OPENMP
    }
#endif
    correspondences_.SetCorrespondenceMatrix(corres_matrix);
    correspondences_.Compress();

    UpdateServer();
}

RegistrationResultCuda TransformEstimationCudaForColoredICP::
ComputeResultsAndTransformation() {
    RegistrationResultCuda result;

    results_.Memset(0);
    TransformEstimationCudaForColoredICPKernelCaller::
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

void TransformEstimationCudaForColoredICP::
ExtractResults(Eigen::Matrix6d &JtJ, Eigen::Vector6d &Jtr, float &rmse) {
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

void TransformEstimationCudaForColoredICP::TransformSourcePointCloud(
    const Eigen::Matrix4d &source_to_target) {
    source_.Transform(source_to_target);
    source_cpu_.Transform(source_to_target);
}
} // cuda
} // open3d
