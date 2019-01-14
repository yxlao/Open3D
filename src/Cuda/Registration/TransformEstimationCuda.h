//
// Created by wei on 1/11/19.
//

#pragma once

#include <Cuda/Common/VectorCuda.h>
#include <Cuda/Container/ArrayCuda.h>
#include <Cuda/Geometry/PointCloudCuda.h>
#include <Cuda/Registration/CorrespondenceSetCuda.h>
#include <Core/Registration/TransformationEstimation.h>


namespace open3d {
namespace cuda {

class RegistrationResultCuda {
public:
    RegistrationResultCuda(const Eigen::Matrix4d &transformation =
    Eigen::Matrix4d::Identity()) : transformation_(transformation),
                                   inlier_rmse_(0.0), fitness_(0.0) {}
    ~RegistrationResultCuda() {}

public:
    Eigen::Matrix4d transformation_;
    CorrespondenceSetCuda correspondence_set_;
    float inlier_rmse_;
    float fitness_;
};

class TransformationEstimationCuda {
public:
    TransformationEstimationCuda() {}
    virtual ~TransformationEstimationCuda() {}

public:
    virtual TransformationEstimationType
    GetTransformationEstimationType() const = 0;

    virtual RegistrationResultCuda ComputeResultsAndTransformation(
        const PointCloudCuda &source,
        const PointCloudCuda &target,
        const CorrespondenceSetCuda &corres) const;
};

class TransformationEstimationPointToPointCuda :
    public TransformationEstimationCuda {

public:
    TransformationEstimationPointToPointCuda(bool with_scaling = false) :
        with_scaling_(with_scaling) {}
    ~TransformationEstimationPointToPointCuda() override {}

public:
    TransformationEstimationType GetTransformationEstimationType()
    const override { return type_; };

    RegistrationResultCuda ComputeResultsAndTransformation(
        const PointCloudCuda &source,
        const PointCloudCuda &target,
        const CorrespondenceSetCuda &corres) const override;

public:
    bool with_scaling_ = false;

private:
    const TransformationEstimationType type_ =
        TransformationEstimationType::PointToPoint;
};

class TransformationEstimationPointToPlaneCuda :
    public TransformationEstimationCuda {

public:
    TransformationEstimationPointToPlaneCuda();
    ~TransformationEstimationPointToPlaneCuda() override {}

public:
    TransformationEstimationType GetTransformationEstimationType()
    const override { return type_; };
    RegistrationResultCuda ComputeResultsAndTransformation(
        const PointCloudCuda &source,
        const PointCloudCuda &target,
        const CorrespondenceSetCuda &corres) const override;

private:
    const TransformationEstimationType type_ =
        TransformationEstimationType::PointToPlane;
};
}
}