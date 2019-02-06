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
#include "../Utils.h"

using namespace open3d;

PoseGraph MatchFragmentsWithOdometry(
    const std::string &base_path,
    const std::vector<std::string> &ply_filenames) {

    PoseGraph pose_graph;
    Eigen::Matrix4d trans_odometry = Eigen::Matrix4d::Identity();
    pose_graph.nodes_.emplace_back(PoseGraphNode(trans_odometry));

    PointCloud source_origin, target_origin;
    for (int s = 0; s < ply_filenames.size() - 1; ++s) {
        ReadPointCloudFromPLY(ply_filenames[s], source_origin);
        auto source = VoxelDownSample(source_origin, kVoxelSize);

        PoseGraph pose_graph_s;
        ReadPoseGraph(GetFragmentPoseGraphName(s, base_path), pose_graph_s);
        auto rbegin = pose_graph_s.nodes_.rbegin();
        Eigen::Matrix4d init_source_to_target = rbegin->pose_.inverse();

        int t = s + 1;
        ReadPointCloudFromPLY(ply_filenames[t], target_origin);
        auto target = VoxelDownSample(target_origin, kVoxelSize);

        cuda::RegistrationCuda registration(
            TransformationEstimationType::ColoredICP);

        registration.Initialize(*source, *target, kVoxelSize * 1.4f,
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

void MatchFragmentsWithLoopClosure(
    const std::string &base_path,
    const std::vector<std::string> &ply_filenames,
    PoseGraph &pose_graph) {

    PointCloud source_origin, target_origin;
    for (int s = 0; s < ply_filenames.size() - 2; ++s) {
        ReadPointCloudFromPLY(ply_filenames[s], source_origin);
        auto source = VoxelDownSample(source_origin, kVoxelSize);
        auto pose_source = pose_graph.nodes_[s].pose_;

        for (int t = s + 2; t < ply_filenames.size(); ++t) {
            auto pose_target = pose_graph.nodes_[t].pose_;

            auto source_to_target_init = pose_target.inverse() * pose_source;
            Eigen::Vector6d delta = TransformMatrix4dToVector6d(
                source_to_target_init);

            ReadPointCloudFromPLY(ply_filenames[t], target_origin);
            auto target = VoxelDownSample(target_origin, kVoxelSize);

            cuda::FastGlobalRegistrationCuda fgr;
            fgr.Initialize(*source, *target);
            cuda::RegistrationResultCuda result;
            for (int iter = 0; iter < 64; ++iter) {
                result = fgr.DoSingleIteration(iter);
            }
            auto trans = result.transformation_;

            /**!!! THIS SHOULD BE REFACTORED !!!**/
            cuda::RegistrationCuda registration(
                TransformationEstimationType::PointToPoint);
            auto source_copy = *source;
            source_copy.Transform(result.transformation_);
            registration.Initialize(source_copy, *target, kVoxelSize * 1.4f);
            registration.transform_source_to_target_ = trans;
            auto information = registration.ComputeInformationMatrix();

            if (result.transformation_.trace() != 4.0
                && information(5, 5) /
                    std::min(source->points_.size(), target->points_.size())
                    >= 0.3) {
                pose_graph.edges_.emplace_back(s, t, trans, information, true);
            }
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

    PoseGraph pose_graph = MatchFragmentsWithOdometry(kBasePath, ply_filenames);
    WritePoseGraph(GetScenePoseGraphName(kBasePath), pose_graph);

//    MatchFragmentsWithLoopClosure(kBasePath, ply_filenames, pose_graph);
    OptimizePoseGraphForScene(kBasePath);
    timer.Stop();
    PrintInfo("RegisterFragments takes %.3f s\n",
              timer.GetDuration() / 1000.0f);
}