//
// Created by wei on 1/11/19.
//

#include "RegistrationCuda.h"
#include <Core/Core.h>

namespace open3d {
namespace cuda {

RegistrationCuda::RegistrationCuda(const TransformationEstimationType &type) {
    if (type == TransformationEstimationType::ColoredICP) {
        estimator_ = std::make_shared<TransformEstimationCudaForColoredICP>();
    } else if (type == TransformationEstimationType::PointToPlane) {
        estimator_ = std::make_shared<TransformEstimationPointToPlaneCuda>();
    } else if (type == TransformationEstimationType::PointToPoint) {
        estimator_ = std::make_shared<TransformEstimationPointToPointCuda>();
    }
}

void RegistrationCuda::Initialize(
    PointCloud &source, PointCloud &target,
    float max_correspondence_distance,
    const Eigen::Matrix<double, 4, 4> &init) {
    estimator_->Initialize(source, target, max_correspondence_distance);

    transform_source_to_target_ = init;
    if (!init.isIdentity()) {
        estimator_->TransformSourcePointCloud(init);
    }
}

RegistrationResultCuda RegistrationCuda::DoSingleIteration(int iter) {
    estimator_->GetCorrespondences();
    auto result = estimator_->ComputeResultsAndTransformation();
    estimator_->TransformSourcePointCloud(result.transformation_);
    transform_source_to_target_ = result.transformation_ *
        transform_source_to_target_;

    PrintInfo("Iteration %d: inlier rmse = %f, fitness = %f\n",
        iter, result.inlier_rmse_, result.fitness_);

    return result;
}

RegistrationResultCuda RegistrationCuda::ComputeICP() {

}
}
}