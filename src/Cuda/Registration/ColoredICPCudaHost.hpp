//
// Created by wei on 1/11/19.
//

#include "ColoredICPCuda.h"

namespace open3d {
namespace cuda {

TransformEstimationForColoredICPCuda::TransformEstimationForColoredICPCuda(
    float lambda_geometric) :
    lambda_geometric_(lambda_geometric) {
    if (lambda_geometric_ < 0 || lambda_geometric_ > 1.0)
        lambda_geometric_ = 0.968f;

    Create();
}

TransformEstimationForColoredICPCuda::~TransformEstimationForColoredICPCuda() {
    Release();
}

void TransformEstimationForColoredICPCuda::Create() {
    device_ = std::make_shared<TransformEstimationForColoredICPCudaDevice>();
    /* 21 JtJ + 6 Jtr + 1 RMSE */
    results_.Create(28);
}

void TransformEstimationForColoredICPCuda::Release() {
    if (device_ != nullptr && device_.use_count() == 1) {
        target_color_gradient_.Release();

        source_.Release();
        target_.Release();
        correspondences_.Release();
        results_.Release();
    }
    device_ = nullptr;
}

void TransformEstimationForColoredICPCuda::UpdateDevice() {
    if (device_ != nullptr) {
        device_->source_ = *source_.device_;
        device_->target_ = *target_.device_;
        device_->target_color_gradient_ = *target_color_gradient_.device_;

        device_->correspondences_ = *correspondences_.device_;

        device_->results_ = *results_.device_;

        device_->lambda_geometric_ = lambda_geometric_;
        device_->sqrt_coeff_G_ = sqrtf(lambda_geometric_);
        device_->sqrt_coeff_I_ = sqrtf(1 - lambda_geometric_);
    }
}

void TransformEstimationForColoredICPCuda::Initialize(
    geometry::PointCloud &source,
    geometry::PointCloud &target, float
    max_correspondence_distance) {

    /** Resize it first -- in super.Initialize, UpdateDevice will be called,
     * where target_color_gradient will be updated **/
    target_color_gradient_.Resize(target.points_.size());

    TransformEstimationCuda::Initialize(
        source, target, max_correspondence_distance);

    ComputeColorGradients(target_cpu_, kdtree_,
        geometry::KDTreeSearchParamHybrid(max_correspondence_distance_ * 2,
            30));
}

void TransformEstimationForColoredICPCuda::ComputeColorGradients(
    geometry::PointCloud &target,
    geometry::KDTreeFlann &kdtree,
    const geometry::KDTreeSearchParamHybrid &search_param) {

    /** Initialize correspondence matrix for neighbors **/
    Eigen::MatrixXi corres_matrix = Eigen::MatrixXi::Constant(
        search_param.max_nn_, target.points_.size(), -1);

    /** OpenMP parallel K-NN search **/
#ifdef _OPENMP
#pragma omp parallel
    {
#endif
#ifdef _OPENMP
#pragma omp for nowait
#endif
        for (int i = 0; i < target.points_.size(); ++i) {
            std::vector<int> indices(search_param.max_nn_);
            std::vector<double> dists(search_param.max_nn_);

            if (kdtree.SearchHybrid(target.points_[i],
                                    search_param.radius_,
                                    search_param.max_nn_,
                                    indices, dists) >= 3) {
                corres_matrix.block(0, i, indices.size(), 1) =
                    Eigen::Map<Eigen::VectorXi>(
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
    TransformEstimationCudaForColoredICPKernelCaller::ComputeColorGradeint(
        *this, corres_for_color_gradient);
}

RegistrationResultCuda TransformEstimationForColoredICPCuda::
ComputeResultsAndTransformation() {
    RegistrationResultCuda result;

    results_.Memset(0);
    TransformEstimationCudaForColoredICPKernelCaller::
    ComputeResultsAndTransformation(*this);

    Eigen::Matrix6d JtJ;
    Eigen::Vector6d Jtr;
    float rmse;
    ExtractResults(JtJ, Jtr, rmse);

    bool is_success;
    Eigen::Matrix4d extrinsic;
    std::tie(is_success, extrinsic) =
        utility::SolveJacobianSystemAndObtainExtrinsicMatrix(JtJ, Jtr);

    int inliers = correspondences_.indices_.size();
    result.fitness_ = float(inliers) / source_.points_.size();
    result.inlier_rmse_ = sqrt(rmse / inliers);
    result.transformation_ = extrinsic;
    // result.correspondences_ = correspondences_;

    return result;
}

} // cuda
} // open3d
