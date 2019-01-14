//
// Created by wei on 1/11/19.
//

#include "RegistrationCuda.h"

namespace open3d {
namespace cuda {

void RegistrationCuda::Initialize(open3d::PointCloud &source,
                                  open3d::PointCloud &target,
                                  const Eigen::Matrix<double, 4, 4> &init) {

}

RegistrationResultCuda RegistrationCuda::ICP(
    PointCloud &source,
    PointCloud &target,
    double max_correspondence_distance,
    const Eigen::Matrix<double, 4, 4> &init,
    const TransformationEstimationCuda &estimation,
    const ICPConvergenceCriteria &criteria) {

    if (max_correspondence_distance <= 0.0) {
        PrintError("Error: Invalid max_correspondence_distance.\n");
        return RegistrationResultCuda(init);
    }
    if (estimation.GetTransformationEstimationType() ==
        TransformationEstimationType::PointToPlane &&
        (!source.HasNormals() || !target.HasNormals())) {
        PrintError(
            "Error: TransformationEstimationPointToPlane requires pre-computed normal vectors.\n");
        return RegistrationResultCuda(init);
    }

    Eigen::Matrix4d transformation = init;
    KDTreeFlann kdtree;
    kdtree.SetGeometry(target);
    PointCloudCuda source_cuda, target_cuda;
    target_cuda.Upload(source);
    source_cuda.Upload(target);

    PointCloud pcd = source;
    if (init.isIdentity() == false) {
        source_cuda.Transform(init);
        pcd = *source_cuda.Download();
    }

    RegistrationResultCuda result;
    GetCorrespondences(pcd, target, kdtree, transformation);
    result = ComputeRegistrationResult(source_cuda, target_cuda,
        max_correspondence_distance, result.correspondence_set_);

    for (int i = 0; i < criteria.max_iteration_; i++) {
        PrintDebug("ICP Iteration #%d: Fitness %.4f, RMSE %.4f\n", i,
                   result.fitness_, result.inlier_rmse_);
        Eigen::Matrix4d update = estimation.ComputeTransformation(
            source_cuda, target_cuda, result.correspondence_set_);
        transformation = update * transformation;
        pcd.Transform(update);

        RegistrationResultCuda backup = result;
        GetCorrespondences(pcd, target, kdtree, transformation);
        result = ComputeRegistrationResult(
            source_cuda, target_cuda, max_correspondence_distance,
            correspondences_);
        if (std::abs(backup.fitness_ - result.fitness_) <
            criteria.relative_fitness_ && std::abs(backup.inlier_rmse_ -
            result.inlier_rmse_) < criteria.relative_rmse_) {
            break;
        }
    }

    return result;
}
}
}