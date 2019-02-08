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

#include "DatasetConfig.h"

using namespace open3d;

namespace MakeFragment {
void MakePoseGraphForFragment(int fragment_id, DatasetConfig &config) {

    cuda::RGBDOdometryCuda<3> odometry;
    odometry.SetIntrinsics(config.intrinsic_);
    odometry.SetParameters(OdometryOption({20, 10, 5},
                                          config.max_depth_diff_,
                                          config.min_depth_,
                                          config.max_depth_), 0.5f);

    cuda::RGBDImageCuda rgbd_source((float) config.min_depth_,
                                    (float) config.max_depth_,
                                    (float) config.depth_factor_);
    cuda::RGBDImageCuda rgbd_target((float) config.min_depth_,
                                    (float) config.max_depth_,
                                    (float) config.depth_factor_);

    const int begin = fragment_id * config.n_frames_per_fragment_;
    const int end = std::min((fragment_id + 1) * config.n_frames_per_fragment_,
                             (int) config.color_files_.size());

    // world_to_source
    Eigen::Matrix4d trans_odometry = Eigen::Matrix4d::Identity();
    PoseGraph pose_graph;
    pose_graph.nodes_.emplace_back(PoseGraphNode(trans_odometry));

    /** TBD: add feature-matching based loop closures **/
    for (int s = begin; s < end - 1; ++s) {
        int t = s + 1;

        Image depth, color;

        ReadImage(config.depth_files_[s], depth);
        ReadImage(config.color_files_[s], color);
        rgbd_source.Upload(depth, color);

        ReadImage(config.depth_files_[t], depth);
        ReadImage(config.color_files_[t], color);
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

    WritePoseGraph(config.GetPoseGraphFileForFragment(fragment_id, false),
                   pose_graph);
}

void OptimizePoseGraphForFragment(int fragment_id,
                                  DatasetConfig &config) {

    PoseGraph pose_graph;
    ReadPoseGraph(config.GetPoseGraphFileForFragment(fragment_id, false),
                  pose_graph);

    GlobalOptimizationConvergenceCriteria criteria;
    GlobalOptimizationOption option(
        config.max_depth_diff_,
        0.25,
        config.preference_loop_closure_odometry_,
        0);
    GlobalOptimizationLevenbergMarquardt optimization_method;
    GlobalOptimization(pose_graph, optimization_method,
                       criteria, option);

    auto pose_graph_prunned = CreatePoseGraphWithoutInvalidEdges(
        pose_graph, option);

    WritePoseGraph(config.GetPoseGraphFileForFragment(fragment_id, true),
                   *pose_graph_prunned);
}

void IntegrateForFragment(int fragment_id, DatasetConfig &config) {

    PoseGraph pose_graph;
    ReadPoseGraph(config.GetPoseGraphFileForFragment(fragment_id, true),
                  pose_graph);

    float voxel_length = config.tsdf_cubic_size_ / 512.0;

    cuda::PinholeCameraIntrinsicCuda intrinsic(config.intrinsic_);
    cuda::TransformCuda trans = cuda::TransformCuda::Identity();
    cuda::ScalableTSDFVolumeCuda<8> tsdf_volume(
        20000, 400000, voxel_length, (float) config.tsdf_truncation_, trans);
    cuda::ScalableMeshVolumeCuda<8> mesher(
        120000, cuda::VertexWithNormalAndColor, 10000000, 20000000);
    cuda::RGBDImageCuda rgbd((float) config.min_depth_,
                             (float) config.max_depth_,
                             (float) config.depth_factor_);

    const int begin = fragment_id * config.n_frames_per_fragment_;
    const int end = std::min((fragment_id + 1) * config.n_frames_per_fragment_,
                             (int) config.color_files_.size());

    for (int i = begin; i < end; ++i) {
        PrintDebug("Integrating frame %d ...\n", i);

        Image depth, color;
        ReadImage(config.depth_files_[i], depth);
        ReadImage(config.color_files_[i], color);
        rgbd.Upload(depth, color);

        /* Use ground truth trajectory */
        Eigen::Matrix4d pose = pose_graph.nodes_[i - begin].pose_;
        trans.FromEigen(pose);

        tsdf_volume.Integrate(rgbd, intrinsic, trans);
    }

    tsdf_volume.GetAllSubvolumes();
    mesher.MarchingCubes(tsdf_volume);
    auto mesh = mesher.mesh().Download();

    PointCloud pcl;
    pcl.points_ = mesh->vertices_;
    pcl.normals_ = mesh->vertex_normals_;
    pcl.colors_ = mesh->vertex_colors_;

    WritePointCloudToPLY(config.GetPlyFileForFragment(fragment_id), pcl);
}

static int Run(DatasetConfig &config) {
    Timer timer;
    timer.Start();

    filesystem::MakeDirectory(config.path_dataset_ + "/fragments_cuda");

    const int num_fragments =
        DIV_CEILING(config.color_files_.size(), config.n_frames_per_fragment_);

    for (int i = 0; i < num_fragments; ++i) {
        PrintInfo("Processing fragment %d / %d\n", i, num_fragments - 1);
        MakePoseGraphForFragment(i, config);
        OptimizePoseGraphForFragment(i, config);
        IntegrateForFragment(i, config);
    }
    timer.Stop();
    PrintInfo("MakeFragment takes %.3f s\n", timer.GetDuration() / 1000.0f);
}
};