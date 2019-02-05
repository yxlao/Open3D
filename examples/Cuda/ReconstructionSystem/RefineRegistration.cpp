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

#include "System.h"

using namespace open3d;

struct Match {
    bool success;
    int s;
    int t;
    Eigen::Matrix4d trans_source_to_target;
    Eigen::Matrix6d information;
};

std::vector<Match> MatchFragments(
    const std::string &base_path,
    const std::vector<std::string> &ply_filenames) {

    PoseGraph pose_graph;
    ReadPoseGraph(GetScenePoseGraphName(base_path, "_optimized"), pose_graph);

    std::vector<Match> matches;
    PointCloud source_origin, target_origin;

    for (auto &edge : pose_graph.edges_) {
        Match match;
        match.s = edge.source_node_id_;
        match.t = edge.target_node_id_;
        PrintDebug("Processing (%d %d)\n", match.s, match.t);

        ReadPointCloudFromPLY(ply_filenames[match.s], source_origin);
        ReadPointCloudFromPLY(ply_filenames[match.t], target_origin);
        auto source = VoxelDownSample(source_origin, 0.05);
        auto target = VoxelDownSample(target_origin, 0.05);

        cuda::RegistrationCuda registration(
            TransformationEstimationType::ColoredICP);
        registration.Initialize(*source, *target, 0.07f,
                                edge.transformation_);
        for (int i = 0; i < 20; ++i) {
            registration.DoSingleIteration(i);
        }
        match.trans_source_to_target =
            registration.transform_source_to_target_;
        match.information = registration.ComputeInformationMatrix();
        match.success = true;
        PrintDebug("Pair (%d %d) odometry computed.\n",
                   match.s, match.t);

        matches.push_back(match);
    }

    return matches;
}

void MakePoseGraphForRefinedScene(
    const std::string &base_path,
    const std::vector<Match> &matches) {
    PoseGraph pose_graph;

    /* world_to_frag0 */
    Eigen::Matrix4d trans_odometry = Eigen::Matrix4d::Identity();
    pose_graph.nodes_.emplace_back(PoseGraphNode(trans_odometry));

    for (auto &match : matches) {
        if (! match.success) continue;
        if (match.t == match.s + 1) {
            /* world_to_fragi */
            trans_odometry = match.trans_source_to_target * trans_odometry;
            auto trans_odometry_inv = trans_odometry.inverse();

            pose_graph.nodes_.emplace_back(PoseGraphNode(trans_odometry_inv));
            pose_graph.edges_.emplace_back(PoseGraphEdge(match.s, match.t,
                match.trans_source_to_target, match.information, false));
        } else {
            pose_graph.edges_.emplace_back(PoseGraphEdge(match.s, match.t,
                match.trans_source_to_target, match.information, true));
        }
    }

    WritePoseGraph(GetScenePoseGraphName(base_path, "_refined"), pose_graph);
}


void OptimizePoseGraphForScene(
    const std::string &base_path) {

    PoseGraph pose_graph;
    ReadPoseGraph(GetScenePoseGraphName(base_path, "_refined"), pose_graph);

    GlobalOptimizationConvergenceCriteria criteria;
    GlobalOptimizationOption option(0.07, 0.25, 5.0, 0);
    GlobalOptimizationLevenbergMarquardt optimization_method;
    GlobalOptimization(pose_graph, optimization_method,
                       criteria, option);

    auto pose_graph_prunned = CreatePoseGraphWithoutInvalidEdges(
        pose_graph, option);

    WritePoseGraph(GetScenePoseGraphName(base_path, "_refined_optimized"),
                   *pose_graph_prunned);
}


int main(int argc, char **argv) {
    SetVerbosityLevel(VerbosityLevel::VerboseDebug);

    std::string kBasePath = "/home/wei/Work/data/stanford/copyroom";
    const int kNumFragments = 55;

    std::string cmd = "mkdir -p " + kBasePath + "/scene_cuda";
    system(cmd.c_str());

    auto ply_filenames = GetFragmentPlyNames(kBasePath, kNumFragments);
    auto matches = MatchFragments(kBasePath, ply_filenames);
    MakePoseGraphForRefinedScene(kBasePath, matches);
    OptimizePoseGraphForScene(kBasePath);
}