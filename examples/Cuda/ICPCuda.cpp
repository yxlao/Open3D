//
// Created by wei on 1/3/19.
//

#include <iostream>
#include <memory>
#include <Eigen/Dense>

#include <Core/Core.h>
#include <IO/IO.h>
#include <Visualization/Visualization.h>

#include <Core/Utility/Timer.h>
#include <Cuda/Registration/RegistrationCuda.h>

#include "Utils.h"

int main(int argc, char **argv) {
    using namespace open3d;
    SetVerbosityLevel(VerbosityLevel::VerboseDebug);

    std::string base_path = "/home/wei/Work/data/stanford/lounge/";
    auto rgbd_filenames = ReadDataAssociation(
        base_path + "data_association.txt");

    PinholeCameraIntrinsic intrinsic = PinholeCameraIntrinsic(
        PinholeCameraIntrinsicParameters::PrimeSenseDefault);

    auto rgbd_source = ReadRGBDImage(base_path + "/" + rgbd_filenames[3].second,
                                     base_path + "/" + rgbd_filenames[3].first,
                                     intrinsic);
    auto rgbd_target = ReadRGBDImage(base_path + "/" + rgbd_filenames[5].second,
                                     base_path + "/" + rgbd_filenames[5].first,
                                     intrinsic);

    auto source_origin = CreatePointCloudFromRGBDImage(*rgbd_source, intrinsic);
    auto target_origin = CreatePointCloudFromRGBDImage(*rgbd_target, intrinsic);

    EstimateNormals(*source_origin);
    EstimateNormals(*target_origin);

    auto source = VoxelDownSample(*source_origin, 0.05);
    auto target = VoxelDownSample(*target_origin, 0.05);

    VisualizeRegistration(*source, *target, Eigen::Matrix4d::Identity());

    { /** PointToPlane **/
        open3d::cuda::RegistrationCuda registration(
            open3d::TransformationEstimationType::PointToPlane);

        registration.Initialize(*source, *target, 0.07f);
        for (int i = 0; i < 30; ++i) {
            auto result = registration.DoSingleIteration(i);
        }
        VisualizeRegistration(*source, *target,
                              registration.transform_source_to_target_);
    }

    { /** PointToPlane **/
        open3d::cuda::RegistrationCuda registration(
            open3d::TransformationEstimationType::PointToPoint);

        registration.Initialize(*source, *target, 0.07f);
        for (int i = 0; i < 30; ++i) {
            auto result = registration.DoSingleIteration(i);
        }
        VisualizeRegistration(*source, *target,
                              registration.transform_source_to_target_);
    }
}