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

std::vector<Match> MatchFragments(
    const std::string &base_path,
    const std::vector<std::string> &ply_filenames) {

    std::vector<Match> matches;

    PointCloud source_origin, target_origin;
    for (int s = 0; s < ply_filenames.size() - 1; ++s) {
        ReadPointCloudFromPLY(ply_filenames[s], source_origin);
        auto source = VoxelDownSample(source_origin, kVoxelSize);

        PoseGraph pose_graph_s;
        ReadPoseGraph(GetFragmentPoseGraphName(s, base_path), pose_graph_s);
        auto rbegin = pose_graph_s.nodes_.rbegin();
        Eigen::Matrix4d init_source_to_target = rbegin->pose_.inverse();

        for (int t = s + 1; t < ply_filenames.size(); ++t) {
            ReadPointCloudFromPLY(ply_filenames[t], target_origin);
            auto target = VoxelDownSample(target_origin, kVoxelSize);

            Match match;
            match.s = s;
            match.t = t;

            if (t == s + 1) {
                cuda::RegistrationCuda registration(
                    TransformationEstimationType::ColoredICP);
                registration.Initialize(*source, *target, kVoxelSize * 1.4f,
                                        init_source_to_target);
                registration.ComputeICP();
                match.trans_source_to_target =
                    registration.transform_source_to_target_;
                match.information = registration.ComputeInformationMatrix();
                match.success = true;
                PrintDebug("Pair (%d %d) odometry computed.\n",
                    match.s, match.t);
            } else {
                cuda::FastGlobalRegistrationCuda fgr;
                fgr.Initialize(*source, *target);

                auto result = fgr.ComputeRegistration();
                match.trans_source_to_target = result.transformation_;

                /**!!! THIS SHOULD BE REFACTORED !!!**/
                cuda::RegistrationCuda registration(
                    TransformationEstimationType::PointToPoint);
                auto source_copy = *source;
                source_copy.Transform(result.transformation_);
                registration.Initialize(source_copy, *target, kVoxelSize * 1.4f);
                registration.transform_source_to_target_ =
                    result.transformation_;
                match.information = registration.ComputeInformationMatrix();

                match.success = match.trans_source_to_target.trace() != 4.0
                    && match.information(5, 5) /
                        std::min(source->points_.size(), target->points_.size())
                        >= 0.3;
                if (match.success) {
                    PrintDebug("Pair (%d %d) registration computed.\n",
                               match.s, match.t);
                } else {
                    PrintDebug("Skip (%d %d) registration.\n",
                               match.s, match.t);
                }
            }

            matches.push_back(match);
        }
    }

    return matches;
}

void MakePoseGraphForScene(
    const std::string &base_path,
    const std::vector<Match> &matches) {
    PoseGraph pose_graph;

    /* world_to_frag_0 */
    Eigen::Matrix4d trans_odometry = Eigen::Matrix4d::Identity();
    pose_graph.nodes_.emplace_back(PoseGraphNode(trans_odometry));

    for (auto &match : matches) {
        if (! match.success) continue;
        if (match.t == match.s + 1) {
            /* world_to_frag_i */
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

    WritePoseGraph(GetScenePoseGraphName(base_path), pose_graph);
}


void OptimizePoseGraphForScene(
    const std::string &base_path) {

    PoseGraph pose_graph;
    ReadPoseGraph(GetScenePoseGraphName(base_path), pose_graph);

    GlobalOptimizationConvergenceCriteria criteria;
    GlobalOptimizationOption option(
        kMaxDepthDiff, 0.25, kPreferenceLoopClosureRegistration, 0);
    GlobalOptimizationLevenbergMarquardt optimization_method;
    GlobalOptimization(pose_graph, optimization_method,
                       criteria, option);

    auto pose_graph_prunned = CreatePoseGraphWithoutInvalidEdges(
        pose_graph, option);

    WritePoseGraph(GetScenePoseGraphName(base_path, "_optimized"),
        *pose_graph_prunned);
}


int main(int argc, char **argv) {
    SetVerbosityLevel(VerbosityLevel::VerboseDebug);

    Timer timer;
    timer.Start();
    std::string cmd = "mkdir -p " + kBasePath + "/scene_cuda";
    system(cmd.c_str());

    auto ply_filenames = GetFragmentPlyNames(kBasePath, kNumFragments);
    auto matches = MatchFragments(kBasePath, ply_filenames);
    MakePoseGraphForScene(kBasePath, matches);
    OptimizePoseGraphForScene(kBasePath);
    timer.Stop();
    PrintInfo("RegisterFragments takes %.3f s\n",
        timer.GetDuration() / 1000.0f);
}