//
// Created by wei on 1/23/19.
//

//
// Created by wei on 1/21/19.
//

#include <Core/Core.h>
#include <IO/IO.h>
#include <Visualization/Visualization.h>
#include <Core/Registration/FastGlobalRegistration.h>
#include <Core/Registration/ColoredICP.h>
#include <iostream>
#include <Cuda/Geometry/NNCuda.h>
#include <Cuda/Registration/FastGlobalRegistrationCuda.h>
#include "Utils.h"

int main(int argc, char **argv) {
    using namespace open3d;

    SetVerbosityLevel(VerbosityLevel::VerboseDebug);

    std::string filepath = "/home/wei/Work/data/stanford/lounge/fragments";
    auto source_origin = CreatePointCloudFromFile(
        filepath + "/fragment_005.ply");
    auto target_origin = CreatePointCloudFromFile(
        filepath + "/fragment_008.ply");

    auto source = VoxelDownSample(*source_origin, 0.05);
    auto target = VoxelDownSample(*target_origin, 0.05);

    EstimateNormals(*source, KDTreeSearchParamHybrid(0.1, 30));
    EstimateNormals(*target, KDTreeSearchParamHybrid(0.1, 30));

    /* Before */
    VisualizeRegistration(*source, *target, Eigen::Matrix4d::Identity());

    cuda::FastGlobalRegistrationCuda fgr;
    fgr.Initialize(*source, *target);
    cuda::RegistrationResultCuda result;
    for (int iter = 0; iter < 64; ++iter) {
        result = fgr.DoSingleIteration(iter);
    }

    /* After */
    VisualizeRegistration(*source, *target, result.transformation_);

    return 0;
}