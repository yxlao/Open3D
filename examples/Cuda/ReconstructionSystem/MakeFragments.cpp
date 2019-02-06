//
// Created by wei on 2/4/19.
//

#include <vector>
#include <string>
#include <sstream>
#include <cmath>
#include <iomanip>

#include <Core/Core.h>
#include <Core/Registration/PoseGraph.h>
#include <Core/Registration/GlobalOptimization.h>

#include <IO/IO.h>

#include <Cuda/Odometry/RGBDOdometryCuda.h>
#include <Cuda/Integration/ScalableTSDFVolumeCuda.h>
#include <Cuda/Integration/ScalableMeshVolumeCuda.h>

#include "../Utils.h"
#include "System.h"

using namespace open3d;

void MakePoseGraphForFragment(
    int fragment_id,
    const std::string &base_path,
    const std::vector<std::pair<std::string, std::string>> &filenames) {

    cuda::RGBDOdometryCuda<3> odometry;
    odometry.SetIntrinsics(PinholeCameraIntrinsic(
        PinholeCameraIntrinsicParameters::PrimeSenseDefault));
    odometry.SetParameters(OdometryOption(
        {20, 10, 5}, kMaxDepthDiff, kDepthMin, kDepthMax),
                           0.5f);

    cuda::RGBDImageCuda rgbd_source(kDepthMin, kDepthMax, kDepthFactor);
    cuda::RGBDImageCuda rgbd_target(kDepthMin, kDepthMax, kDepthFactor);

    const int begin = fragment_id * kFramesPerFragment;
    const int end = std::min((fragment_id + 1) * kFramesPerFragment,
                             (int) filenames.size());

    // world_to_source
    Eigen::Matrix4d trans_odometry = Eigen::Matrix4d::Identity();
    PoseGraph pose_graph;
    pose_graph.nodes_.emplace_back(PoseGraphNode(trans_odometry));

    /** TBD: add feature-matching based loop closures **/
    for (int s = begin; s < end - 1; ++s) {
        int t = s + 1;

        Image depth, color;

        ReadImage(base_path + "/" + filenames[s].first, depth);
        ReadImage(base_path + "/" + filenames[s].second, color);
        rgbd_source.Upload(depth, color);

        ReadImage(base_path + "/" + filenames[t].first, depth);
        ReadImage(base_path + "/" + filenames[t].second, color);
        rgbd_target.Upload(depth, color);

        odometry.transform_source_to_target_ = Eigen::Matrix4d::Identity();
        odometry.Initialize(rgbd_source, rgbd_target);
        odometry.ComputeMultiScale();

        Eigen::Matrix4d trans = odometry.transform_source_to_target_;
        Eigen::Matrix6d information = odometry.ComputeInformationMatrix();

        // source_to_target * world_to_source = world_to_target
        trans_odometry = trans * trans_odometry;

        // target_to_world
        Eigen::Matrix4d trans_odometry_inv = trans_odometry.inverse();

        pose_graph.nodes_.emplace_back(PoseGraphNode(trans_odometry_inv));
        pose_graph.edges_.emplace_back(PoseGraphEdge(
            s - begin, t - begin, trans, information, false));
    }

    WritePoseGraph(GetFragmentPoseGraphName(fragment_id, base_path),
                   pose_graph);
}

void OptimizePoseGraphForFragment(
    int fragment_id, const std::string &base_path) {

    PoseGraph pose_graph;
    ReadPoseGraph(GetFragmentPoseGraphName(fragment_id, base_path), pose_graph);

    GlobalOptimizationConvergenceCriteria criteria;
    GlobalOptimizationOption option(
        kMaxDepthDiff, 0.25, kPreferenceLoopClosureOdometry, 0);
    GlobalOptimizationLevenbergMarquardt optimization_method;
    GlobalOptimization(pose_graph, optimization_method,
                       criteria, option);

    auto pose_graph_prunned = CreatePoseGraphWithoutInvalidEdges(
        pose_graph, option);

    WritePoseGraph(GetFragmentPoseGraphName(
        fragment_id, base_path, "optimized_"),
                   *pose_graph_prunned);
}

void IntegrateForFragment(
    int fragment_id,
    const std::string &base_path,
    const std::vector<std::pair<std::string, std::string>> &filenames) {

    PoseGraph pose_graph;
    ReadPoseGraph(GetFragmentPoseGraphName(
        fragment_id, base_path, "optimized_"),
                  pose_graph);

    cuda::PinholeCameraIntrinsicCuda intrinsics(
        PinholeCameraIntrinsicParameters::PrimeSenseDefault);

    float voxel_length = kCubicSize / 512.0f;

    cuda::TransformCuda trans = cuda::TransformCuda::Identity();
    cuda::ScalableTSDFVolumeCuda<8> tsdf_volume(
        20000, 400000, voxel_length, kTSDFTruncation, trans);
    cuda::ScalableMeshVolumeCuda<8> mesher(
        120000, cuda::VertexWithNormalAndColor, 10000000, 20000000);
    cuda::RGBDImageCuda rgbd(kDepthMin, kDepthMax, kDepthFactor);

    const int begin = fragment_id * kFramesPerFragment;
    const int end = std::min((fragment_id + 1) * kFramesPerFragment,
                             (int) filenames.size());

    for (int i = begin; i < end; ++i) {
        PrintDebug("Integrating frame %d ...\n", i);

        Image depth, color;
        ReadImage(base_path + "/" + filenames[i].first, depth);
        ReadImage(base_path + "/" + filenames[i].second, color);
        rgbd.Upload(depth, color);

        /* Use ground truth trajectory */
        Eigen::Matrix4d pose = pose_graph.nodes_[i - begin].pose_;
        trans.FromEigen(pose);

        tsdf_volume.Integrate(rgbd, intrinsics, trans);
    }

    tsdf_volume.GetAllSubvolumes();
    mesher.MarchingCubes(tsdf_volume);
    auto mesh = mesher.mesh().Download();

    PointCloud pcl;
    pcl.points_ = mesh->vertices_;
    pcl.normals_ = mesh->vertex_normals_;
    pcl.colors_ = mesh->vertex_colors_;
    WritePointCloudToPLY(GetFragmentPlyName(fragment_id, base_path), pcl);
}

int main(int argc, char **argv) {
    SetVerbosityLevel(VerbosityLevel::VerboseDebug);

    Timer timer;
    timer.Start();

    auto rgbd_filenames = ReadDataAssociation(
        kBasePath + "/data_association.txt");

    std::string cmd = "mkdir -p " + kBasePath + "/fragments_cuda";
    system(cmd.c_str());

    const int num_fragments =
        DIV_CEILING(rgbd_filenames.size(), kFramesPerFragment);


    for (int i = 0; i < num_fragments; ++i) {
        PrintInfo("Processing fragment %d / %d\n", i, num_fragments - 1);
        MakePoseGraphForFragment(i, kBasePath, rgbd_filenames);
        OptimizePoseGraphForFragment(i, kBasePath);
        IntegrateForFragment(i, kBasePath, rgbd_filenames);
    }
    timer.Stop();
    PrintInfo("MakeFragment takes %.3f s\n", timer.GetDuration() / 1000.0f);
}