//
// Created by wei on 1/24/19.
//

#include <Core/Core.h>
#include <IO/IO.h>
#include <Visualization/Visualization.h>
#include <Core/Registration/FastGlobalRegistration.h>
#include <Core/Registration/ColoredICP.h>
#include <iostream>
#include <Cuda/Geometry/NNCuda.h>
#include <Cuda/Registration/FeatureExtractorCuda.h>
#include "ReadDataAssociation.h"

using namespace open3d;
std::shared_ptr<Feature> PreprocessPointCloud(PointCloud &pcd) {
    EstimateNormals(pcd, open3d::KDTreeSearchParamHybrid(0.1, 30));
    auto pcd_fpfh = ComputeFPFHFeature(
        pcd, open3d::KDTreeSearchParamHybrid(0.25, 100));
    return pcd_fpfh;
}

void VisualizeRegistration(const open3d::PointCloud &source,
                           const open3d::PointCloud &target,
                           const Eigen::Matrix4d &Transformation) {
    using namespace open3d;
    std::shared_ptr<PointCloud> source_transformed_ptr(new PointCloud);
    std::shared_ptr<PointCloud> target_ptr(new PointCloud);
    *source_transformed_ptr = source;
    *target_ptr = target;
    source_transformed_ptr->Transform(Transformation);
    DrawGeometries({source_transformed_ptr, target_ptr}, "Registration result");
}

int main(int argc, char **argv) {
    open3d::SetVerbosityLevel(VerbosityLevel::VerboseDebug);
    std::string filepath = "/home/wei/Work/data/stanford/lounge/fragments";
    auto source_origin = CreatePointCloudFromFile(
        filepath + "/fragment_000.ply");
    auto target_origin = CreatePointCloudFromFile(
        filepath + "/fragment_004.ply");

    auto source = open3d::VoxelDownSample(*source_origin, 0.05);
    auto target = open3d::VoxelDownSample(*target_origin, 0.05);

    auto source_feature = PreprocessPointCloud(*source);
    auto target_feature = PreprocessPointCloud(*target);

    std::vector<std::reference_wrapper<const CorrespondenceChecker>>
        correspondence_checker;
    auto correspondence_checker_edge_length =
        CorrespondenceCheckerBasedOnEdgeLength(0.9);
    auto correspondence_checker_distance =
        CorrespondenceCheckerBasedOnDistance(0.075);
    auto correspondence_checker_normal =
        CorrespondenceCheckerBasedOnNormal(0.52359878);

    correspondence_checker.push_back(correspondence_checker_edge_length);
    correspondence_checker.push_back(correspondence_checker_distance);
    correspondence_checker.push_back(correspondence_checker_normal);

    for (int i = 0; i < 100; ++i) {
        auto registration_result = RegistrationRANSACBasedOnFeatureMatching(
            *source, *target, *source_feature, *target_feature, 0.075,
            TransformationEstimationPointToPoint(false), 4,
            correspondence_checker, RANSACConvergenceCriteria(4000000, 1000));
        VisualizeRegistration(*source, *target,
                              registration_result.transformation_);
    }
    return 0;
}