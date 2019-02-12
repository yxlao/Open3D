//
// Created by wei on 2/4/19.
//

#include <vector>
#include <string>
#include <Core/Core.h>
#include <IO/IO.h>
#include <Visualization/Visualization.h>

#include <Cuda/Registration/RegistrationCuda.h>
#include <Cuda/Registration/ColoredICPCuda.h>
#include <Cuda/Registration/FastGlobalRegistrationCuda.h>

#include <Core/Registration/PoseGraph.h>
#include <Core/Registration/GlobalOptimization.h>

#include "DatasetConfig.h"

using namespace open3d;

std::vector<Match> MatchFragments(DatasetConfig &config) {
    std::vector<Match> matches;

    for (int s = 55; s < config.thumbnail_fragment_files_.size() - 1; ++s) {
        auto source = CreatePointCloudFromFile(config.thumbnail_fragment_files_[s]);

        PoseGraph pose_graph_s;
        ReadPoseGraph(config.GetPoseGraphFileForFragment(s, true),
                      pose_graph_s);

        auto rbegin = pose_graph_s.nodes_.rbegin();
        Eigen::Matrix4d init_source_to_target = rbegin->pose_.inverse();

        int t = s + 1;
        auto target = CreatePointCloudFromFile(config.thumbnail_fragment_files_[t]);

        Match match;
        match.s = s;
        match.t = t;

        cuda::RegistrationCuda registration(
            TransformationEstimationType::ColoredICP);
        registration.Initialize(*source, *target,
                                (float) config.voxel_size_ * 1.4f,
                                init_source_to_target);
        registration.ComputeICP();
        match.trans_source_to_target =
            registration.transform_source_to_target_;
        match.information = registration.ComputeInformationMatrix();
        match.success = true;
        PrintDebug("Pair (%d %d) odometry computed.\n", match.s, match.t);
        matches.push_back(match);

//        DrawGeometries({source}, "source");
//        DrawGeometries({target}, "target");
        DrawGeometries({source, target});
        source->Transform(match.trans_source_to_target);
        DrawGeometries({source, target});
    }

    return matches;
}

int main(int argc, char **argv) {
    DatasetConfig config;

    std::string config_path = argc > 1 ? argv[1] :
                              "/home/wei/Work/projects/dense_mapping/Open3D/examples/Cuda"
                              "/ReconstructionSystem/config/apartment.json";

    bool is_success = ReadIJsonConvertible(config_path, config);
    if (!is_success) return 1;

    SetVerbosityLevel(VerbosityLevel::VerboseDebug);

    Timer timer;
    timer.Start();
    filesystem::MakeDirectory(config.path_dataset_ + "/scene_cuda");

    is_success = config.GetThumbnailFragmentFiles();
    if (! is_success) {
        PrintError("Unable to get fragment files\n");
        return -1;
    }

    auto matches = MatchFragments(config);
    timer.Stop();

    PrintInfo("RegisterFragments takes %.3f s\n",
              timer.GetDuration() / 1000.0f);
    return 0;
}