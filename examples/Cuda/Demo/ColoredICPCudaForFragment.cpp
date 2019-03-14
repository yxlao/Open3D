//
// Created by wei on 1/3/19.
//

#include <Eigen/Dense>

#include <Core/Core.h>
#include <IO/IO.h>
#include <Visualization/Visualization.h>

#include <Cuda/Registration/ColoredICPCuda.h>
#include <Cuda/Registration/RegistrationCuda.h>

#include "examples/Cuda/Utils.h"

int main(int argc, char **argv) {
    using namespace open3d;
    SetVerbosityLevel(VerbosityLevel::VerboseDebug);

    std::string filepath = "/media/wei/Data/data/indoor_lidar_rgbd/apartment"
                           "/fragments_cuda";
    auto source_origin = CreatePointCloudFromFile(
        filepath + "/fragment_203.ply");
    auto target_origin = CreatePointCloudFromFile(
        filepath + "/fragment_204.ply");

    auto source = VoxelDownSample(*source_origin, 0.05);
    auto target = VoxelDownSample(*target_origin, 0.05);

    /* Before */
    VisualizeRegistration(*source, *target, Eigen::Matrix4d::Identity());

    Eigen::Matrix4d init_source_to_target = Eigen::Matrix4d::Identity();
    init_source_to_target <<
    0.945072,  0.202421, -0.256641, 0.0314436,
        -0.282933,  0.899767, -0.332217, -0.103088,
    0.163669,  0.386582,  0.907616, -0.106229,
        -0,         0,        -0,         1;

    cuda::RegistrationCuda registration(
        TransformationEstimationType::ColoredICP);
    registration.Initialize(*source, *target, 0.07f, init_source_to_target);

    registration.ComputeICP();
    VisualizeRegistration(*source_origin, *target_origin,
                          registration.transform_source_to_target_);
}