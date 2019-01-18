//
// Created by wei on 1/11/19.
//

#pragma once

#include <Cuda/Common/LinearAlgebraCuda.h>
#include <Cuda/Container/ArrayCuda.h>
#include <Cuda/Geometry/PointCloudCuda.h>
#include <Cuda/Registration/CorrespondenceSetCuda.h>
#include <Core/Registration/TransformationEstimation.h>
#include <Core/Geometry/KDTreeFlann.h>

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
    CorrespondenceSetCuda correspondences_;
    float inlier_rmse_;
    float fitness_;
};

class TransformEstimationCuda {
public:
    TransformEstimationCuda() {}
    virtual ~TransformEstimationCuda() {}

public:
    virtual TransformationEstimationType
    GetTransformationEstimationType() const = 0;
    virtual void UpdateServer() = 0;
    virtual RegistrationResultCuda ComputeResultsAndTransformation() = 0;

    virtual void Initialize(
        PointCloud &source, PointCloud &target,
        float max_correspondence_distance);
    virtual void GetCorrespondences();
    virtual void TransformSourcePointCloud(
        const Eigen::Matrix4d &source_to_target);

    virtual void ExtractResults(
        Eigen::Matrix6d &JtJ, Eigen::Vector6d &Jtr, float &rmse);

public:
    /** For GPU **/
    PointCloudCuda source_;
    PointCloudCuda target_;

    /** 1-nn, source x 1 (in target) **/
    CorrespondenceSetCuda correspondences_;
    float max_correspondence_distance_;

    /** For CPU NN search **/
    PointCloud source_cpu_;
    PointCloud target_cpu_;
    KDTreeFlann kdtree_;
    Eigen::Matrix<int, -1, -1, Eigen::RowMajor> corres_matrix_;

    /** Build linear system **/
    ArrayCuda<float> results_;
};
//
//class TransformEstimationPointToPointCuda : public TransformEstimationCuda {
//
//public:
//    TransformEstimationPointToPointCuda(bool with_scaling = false) :
//        with_scaling_(with_scaling) {}
//    ~TransformEstimationPointToPointCuda() override {}
//
//public:
//    TransformationEstimationType GetTransformationEstimationType()
//    const override { return type_; };
//
//    void Initialize(
//        PointCloud &source, PointCloud &target,
//        float max_correspondence_distance) override;
//
//    void GetCorrespondences() override;
//    RegistrationResultCuda ComputeResultsAndTransformation() override;
//
//    void TransformSourcePointCloud(
//        const Eigen::Matrix4d &source_to_target) override;
//
//
//public:
//    bool with_scaling_ = false;
//
//private:
//    const TransformationEstimationType type_ =
//        TransformationEstimationType::PointToPoint;
//};
//
//class TransformEstimationPointToPlaneCuda : public TransformEstimationCuda {
//
//public:
//    TransformEstimationPointToPlaneCuda();
//    ~TransformEstimationPointToPlaneCuda() override {}
//
//public:
//    TransformationEstimationType GetTransformationEstimationType()
//    const override { return type_; };
//
//    void Initialize(
//        PointCloud &source, PointCloud &target,
//        float max_correspondence_distance) override;
//
//    void GetCorrespondences() override;
//    RegistrationResultCuda ComputeResultsAndTransformation() override;
//
//    void TransformSourcePointCloud(
//        const Eigen::Matrix4d &source_to_target) override;
//
//private:
//    const TransformationEstimationType type_ =
//        TransformationEstimationType::PointToPlane;
//};
}
}