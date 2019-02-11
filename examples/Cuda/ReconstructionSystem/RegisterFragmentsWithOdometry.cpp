//
// Created by wei on 2/4/19.
//

#include <vector>
#include <string>
#include <Core/Core.h>
#include <IO/IO.h>

#include <Cuda/Registration/RegistrationCuda.h>
#include <Cuda/Registration/ColoredICPCuda.h>
#include <Cuda/Registration/FastGlobalRegistrationCuda.h>

#include <Core/Registration/PoseGraph.h>
#include <Core/Registration/GlobalOptimization.h>

#include "DatasetConfig.h"

using namespace open3d;

PoseGraph MatchFragmentsWithOdometry(DatasetConfig &config) {

    PoseGraph pose_graph;
    Eigen::Matrix4d trans_odometry = Eigen::Matrix4d::Identity();
    pose_graph.nodes_.emplace_back(PoseGraphNode(trans_odometry));

    for (int s = 0; s < config.fragment_files_.size() - 1; ++s) {
        auto source = CreatePointCloudFromFile(config.fragment_files_[s]);

        PoseGraph pose_graph_s;
        ReadPoseGraph(config.GetPoseGraphFileForFragment(s, true),
            pose_graph_s);
        auto rbegin = pose_graph_s.nodes_.rbegin();
        Eigen::Matrix4d init_source_to_target = rbegin->pose_.inverse();

        int t = s + 1;
        auto target = CreatePointCloudFromFile(config.fragment_files_[t]);

        cuda::RegistrationCuda registration(
            TransformationEstimationType::ColoredICP);

        registration.Initialize(*source, *target, config.voxel_size_ * 1.4f,
                                init_source_to_target);
        registration.ComputeICP();

        trans_odometry = registration.transform_source_to_target_ *
            trans_odometry;
        auto trans_odometry_inv = trans_odometry.inverse();

        pose_graph.nodes_.emplace_back(PoseGraphNode(trans_odometry_inv));
        pose_graph.edges_.emplace_back(PoseGraphEdge(
            s, t, registration.transform_source_to_target_,
            registration.ComputeInformationMatrix(), false));

        PrintDebug("Pair (%d %d) odometry computed.\n", s, t);
    }

    return pose_graph;
}

int main(int argc, char **argv) {
    SetVerbosityLevel(VerbosityLevel::VerboseDebug);

    DatasetConfig config;

    std::string config_path = argc > 1 ? argv[1] :
                              "/home/wei/Work/projects/dense_mapping/Open3D/examples/Cuda"
                              "/ReconstructionSystem/config/nsh.json";

    bool is_success = ReadIJsonConvertible(config_path, config);
    if (!is_success) return 1;

    config.GetFragmentFiles();
    auto pose_graph = MatchFragmentsWithOdometry(config);
    WritePoseGraph(config.GetPoseGraphFileForScene(false), pose_graph);
}