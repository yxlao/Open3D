//
// Created by wei on 1/11/19.
//

#include "RegistrationCuda.h"

namespace open3d {
namespace cuda {

/* Preparation */
RegistrationCuda::RegistrationCuda(
    const registration::TransformationEstimationType &type) {
    Create(type);
}

RegistrationCuda::~RegistrationCuda() {
    Release();
}

void RegistrationCuda::Create(
    const registration::TransformationEstimationType &type) {
    type_ = type;
    lambda_geometric_ = 0.968;

    device_ = std::make_shared<RegistrationCudaDevice>();
    results_.Create(28);
}

void RegistrationCuda::Release() {
    if (device_ != nullptr && device_.use_count() == 1) {
        source_.Release();
        target_.Release();
        correspondences_.Release();
        results_.Release();

        if (type_ == registration::TransformationEstimationType::ColoredICP) {
            target_color_gradient_.Release();
        }
    }

    device_ = nullptr;
}

void RegistrationCuda::UpdateDevice() {
    if (device_ != nullptr) {
        device_->source_ = *source_.device_;
        device_->target_ = *target_.device_;

        device_->correspondences_ = *correspondences_.device_;

        device_->results_ = *results_.device_;

        if (type_ == registration::TransformationEstimationType::ColoredICP) {
            device_->target_color_gradient_ = *target_color_gradient_.device_;
            device_->sqrt_coeff_G_ = sqrtf(lambda_geometric_);
            device_->sqrt_coeff_I_ = sqrtf(1 - lambda_geometric_);
        }
    }
}

void RegistrationCuda::Initialize(
    geometry::PointCloud &source, geometry::PointCloud &target,
    float max_correspondence_distance,
    const Eigen::Matrix<double, 4, 4> &init) {

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

    transform_source_to_target_ = init;
    if (!init.isIdentity()) {
        TransformSourcePointCloud(init);
    }

    if (type_ == registration::TransformationEstimationType::ColoredICP) {
        target_color_gradient_.Resize(target.points_.size());
    }
    UpdateDevice();

    if (type_ == registration::TransformationEstimationType::ColoredICP) {
        ComputeColorGradients(target_cpu_, kdtree_,
                              geometry::KDTreeSearchParamHybrid(
                                  max_correspondence_distance_ * 2, 30));
    }
}

/* High-level API */
RegistrationResultCuda RegistrationCuda::ComputeICP(int iter) {
    assert(iter > 0);

    auto result = DoSingleIteration(0);
    float prev_fitness = result.fitness_;
    float prev_rmse = result.inlier_rmse_;

    for (int i = 1; i < iter; ++i) {
        result = DoSingleIteration(i);

        if (std::abs(prev_fitness - result.fitness_) < 1e-6
            && std::abs(prev_rmse - result.inlier_rmse_) < 1e-6) {
            return result;
        }

        prev_fitness = result.fitness_;
        prev_rmse = result.inlier_rmse_;
    }

    return result;
}

Eigen::Matrix6d RegistrationCuda::ComputeInformationMatrix() {
    /** Point clouds should have been transformed during registration **/
    GetCorrespondences();
    if (correspondences_.indices_.size() < 10) {
        return Eigen::Matrix6d::Identity();
    }

    RegistrationResultCuda result;

    results_.Memset(0);
    RegistrationCudaKernelCaller::ComputeInformationMatrix(*this);

    Eigen::Matrix6d JtJ;
    Eigen::Vector6d Jtr; // dummy
    float rmse;          // dummy
    ExtractResults(JtJ, Jtr, rmse);

    return Eigen::Matrix6d::Identity() + JtJ;
}

Eigen::Matrix6d RegistrationCuda::ComputeInformationMatrix(
    geometry::PointCloud &source,
    geometry::PointCloud &target,
    float max_correspondence_distance,
    const Eigen::Matrix4d &init) {

    RegistrationCuda registration(
        registration::TransformationEstimationType::PointToPoint);
    registration.Initialize(source, target, max_correspondence_distance, init);
    return registration.ComputeInformationMatrix();
}

/* ICP */
RegistrationResultCuda RegistrationCuda::DoSingleIteration(int iter) {
    RegistrationResultCuda delta;
    delta.transformation_ = Eigen::Matrix4d::Identity();
    delta.fitness_ = delta.inlier_rmse_ = 0;

    GetCorrespondences();

    if (correspondences_.indices_.size() < 10) {
        utility::PrintError("Insufficient correspondences: %d\n",
                            correspondences_.indices_.size());
        return delta;
    }

    delta = BuildAndSolveLinearSystem();

    utility::PrintDebug("Iteration %d: inlier rmse = %f, fitness = %f\n",
                        iter, delta.inlier_rmse_, delta.fitness_);

    TransformSourcePointCloud(delta.transformation_);
    transform_source_to_target_ = delta.transformation_ *
        transform_source_to_target_;

    return delta;
}

void RegistrationCuda::GetCorrespondences() {
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

    UpdateDevice();
}

void RegistrationCuda::TransformSourcePointCloud(
    const Eigen::Matrix4d &source_to_target) {
    source_.Transform(source_to_target);
    source_cpu_.Transform(source_to_target);
}

RegistrationResultCuda RegistrationCuda::BuildAndSolveLinearSystem() {
    RegistrationResultCuda result;

    if (type_ == registration::TransformationEstimationType::PointToPoint) {
        return Umeyama();
    }

    results_.Memset(0);
    if (type_ == registration::TransformationEstimationType::ColoredICP) {
        RegistrationCudaKernelCaller
        ::BuildLinearSystemForColoredICP(*this);
    } else if (type_ == registration::TransformationEstimationType::PointToPlane) {
        RegistrationCudaKernelCaller
        ::BuildLinearSystemForPointToPlaneICP(*this);
    }

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
    result.inlier_rmse_ = sqrtf(rmse / inliers);
    result.transformation_ = extrinsic;

    return result;
}

void RegistrationCuda::ExtractResults(
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

void RegistrationCuda::ComputeColorGradients(
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
    RegistrationCudaKernelCaller::ComputeColorGradeint(
        *this, corres_for_color_gradient);
}

RegistrationResultCuda RegistrationCuda::Umeyama() {
    RegistrationResultCuda result;

    int inliers = correspondences_.indices_.size();

    /** Pass 1: sum reduction means **/
    results_.Memset(0);
    RegistrationCudaKernelCaller::ComputeSumForPointToPointICP(*this);
    std::vector<float> downloaded_result = results_.DownloadAll();

    Vector3f mean_source, mean_traget;
    for (int i = 0; i < 3; ++i) {
        mean_source(i) = downloaded_result[i + 0];
        mean_traget(i) = downloaded_result[i + 3];
    }
    mean_source /= inliers;
    mean_traget /= inliers;

    /** Pass 2: sum reduction Sigma **/
    results_.Memset(0);
    RegistrationCudaKernelCaller::BuildLinearSystemForPointToPointICP(
        *this, mean_source, mean_traget);
    downloaded_result = results_.DownloadAll();

    Eigen::Matrix3d Sigma;
    float source_sigma2, rmse;
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            Sigma(i, j) = downloaded_result[i * 3 + j];
        }
    }
    source_sigma2 = downloaded_result[9];
    rmse = downloaded_result[10];

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
    double scale = 1.0;
    // with_scaling_ ? (1.0 / source_sigma2) * (S * d).sum() : 1.0;
    Eigen::Vector3d t = mean_traget.ToEigen() - scale * R * mean_source.ToEigen();

    Eigen::Matrix4d extrinsic = Eigen::Matrix4d::Identity();
    extrinsic.block<3, 3>(0, 0) = scale * R;
    extrinsic.block<3, 1>(0, 3) = t;

    result.fitness_ = float(inliers) / source_.points_.size();
    result.inlier_rmse_ = sqrtf(rmse / inliers);
    result.transformation_ = extrinsic;

    return result;
}
}
}