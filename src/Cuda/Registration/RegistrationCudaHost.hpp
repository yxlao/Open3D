//
// Created by wei on 1/11/19.
//

#include "RegistrationCuda.h"

namespace open3d {
namespace cuda {

RegistrationCuda::RegistrationCuda(
    const registration::TransformationEstimationType &type) {
    if (type == registration::TransformationEstimationType::ColoredICP) {
        estimator_ = std::make_shared<TransformEstimationForColoredICPCuda>();
    } else if (type == registration::TransformationEstimationType
    ::PointToPlane) {
        estimator_ = std::make_shared<TransformEstimationPointToPlaneCuda>();
    } else if (type == registration::TransformationEstimationType
    ::PointToPoint) {
        estimator_ = std::make_shared<TransformEstimationPointToPointCuda>();
    }
}

void RegistrationCuda::Initialize(
    geometry::PointCloud &source, geometry::PointCloud &target,
    float max_correspondence_distance,
    const Eigen::Matrix<double, 4, 4> &init) {
    estimator_->Initialize(source, target, max_correspondence_distance);

    transform_source_to_target_ = init;
    if (!init.isIdentity()) {
        estimator_->TransformSourcePointCloud(init);
    }
}

RegistrationResultCuda RegistrationCuda::DoSingleIteration(int iter) {
    RegistrationResultCuda result;
    result.transformation_ = Eigen::Matrix4d::Identity();
    result.fitness_ = result.inlier_rmse_ = 0;

    estimator_->GetCorrespondences();

    if (estimator_->correspondences_.indices_.size() < 10) {
        utility::PrintError("Insufficient correspondences: %d\n",
                   estimator_->correspondences_.indices_.size());
        return result;
    }

    result = estimator_->ComputeResultsAndTransformation();

    utility::PrintDebug("Iteration %d: inlier rmse = %f, fitness = %f\n",
               iter, result.inlier_rmse_, result.fitness_);

    estimator_->TransformSourcePointCloud(result.transformation_);
    transform_source_to_target_ = result.transformation_ *
        transform_source_to_target_;

    return result;
}

Eigen::Matrix6d RegistrationCuda::ComputeInformationMatrix() {
    /** Point clouds should have been transformed during registration **/
    estimator_->GetCorrespondences();
    if (estimator_->correspondences_.indices_.size() < 10) {
        return Eigen::Matrix6d::Identity();
    }

    return estimator_->ComputeInformationMatrix();
}

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
}
}