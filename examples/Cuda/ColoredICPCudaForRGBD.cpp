//
// Created by wei on 1/3/19.
//

#include <Eigen/Dense>

#include <Core/Core.h>
#include <IO/IO.h>
#include <Visualization/Visualization.h>

#include <Cuda/Registration/ColoredICPCuda.h>
#include <Cuda/Registration/RegistrationCuda.h>

#include "Utils.h"

int main(int argc, char **argv) {
    using namespace open3d;
    SetVerbosityLevel(VerbosityLevel::VerboseDebug);

    std::string base_path = "/home/wei/Work/data/stanford/lounge";
    auto rgbd_filenames = ReadDataAssociation(
        base_path + "/data_association.txt");

    PinholeCameraIntrinsic intrinsic = PinholeCameraIntrinsic(
        PinholeCameraIntrinsicParameters::PrimeSenseDefault);

    auto rgbd_source = ReadRGBDImage(
        base_path + "/" + rgbd_filenames[3].second,
        base_path + "/" + rgbd_filenames[3].first,
        intrinsic);
    auto rgbd_target = ReadRGBDImage(
        base_path + "/" + rgbd_filenames[5].second,
        base_path + "/" + rgbd_filenames[5].first,
        intrinsic);

    auto source_origin = CreatePointCloudFromRGBDImage(
        *rgbd_source, intrinsic);
    auto target_origin = CreatePointCloudFromRGBDImage(
        *rgbd_target, intrinsic);

    auto source = VoxelDownSample(*source_origin, 0.05);
    auto target = VoxelDownSample(*target_origin, 0.05);

    EstimateNormals(*source_origin);
    EstimateNormals(*target_origin);

    /* Before */
    VisualizeRegistration(*source, *target,
        Eigen::Matrix4d::Identity());

    cuda::RegistrationCuda registration(
        TransformationEstimationType::ColoredICP);
    registration.Initialize(*source, *target, 0.07f);
    auto result = registration.ComputeICP();

    /* After */
    VisualizeRegistration(*source_origin, *target_origin,
        registration.transform_source_to_target_);
    return 0;
}