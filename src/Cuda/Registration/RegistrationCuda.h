//
// Created by wei on 1/10/19.
//

#pragma once

#include <Core/Registration/Registration.h>
#include <Core/Geometry/PointCloud.h>
#include <Core/Geometry/KDTreeFlann.h>

#include <Cuda/Container/ArrayCuda.h>
#include <Cuda/Geometry/PointCloudCuda.h>
#include <Cuda/Registration/TransformEstimationCuda.h>

namespace open3d {

namespace cuda {
class ICPConvergenceCriteriaCuda {
public:
    ICPConvergenceCriteriaCuda(float relative_fitness = 1e-6f,
                               float relative_rmse = 1e-6f,
                               int max_iteration = 30) :
        relative_fitness_(relative_fitness), relative_rmse_(relative_rmse),
        max_iteration_(max_iteration) {}
    ~ICPConvergenceCriteriaCuda() {}

public:
    float relative_fitness_;
    float relative_rmse_;
    int max_iteration_;
};

class RegistrationCudaDevice {
public:
    PointCloudCudaDevice source_;
    PointCloudCudaDevice target_;
    CorrespondenceSetCuda correspondences_;
};

class RegistrationCuda {
private:
    std::shared_ptr<RegistrationCudaDevice> server_ = nullptr;

    /** Used for knn search **/
    PointCloud source_;
    PointCloud target_;
    KDTreeFlann target_kdtree_;

    PointCloudCuda source_cuda_;
    PointCloudCuda target_cuda_;

    CorrespondenceSetCuda correspondences_;
    Eigen::Matrix4d transform_source_to_target_;

public:
    void Initialize(
        PointCloud &source, PointCloud &target,
        const Eigen::Matrix4d &init = Eigen::Matrix4d::Identity());

    void SetParameters(
        float max_correspondence_distance,
        const TransformationEstimationCuda &estimation =
        TransformationEstimationCuda(),
        const ICPConvergenceCriteria &criteria = ICPConvergenceCriteria());

    RegistrationResultCuda DoSingleIteration(int iter);
    RegistrationResultCuda ComputeICP();

public:
    /** Iterations **/
    /* CPU */
    void GetCorrespondences();

    /* GPU */
    RegistrationResultCuda ComputeRegistrationResult();
    /* ... and estimation.ComputeTransform */

};
}
}


