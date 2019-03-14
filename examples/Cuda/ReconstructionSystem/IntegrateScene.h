//
// Created by wei on 2/4/19.
//

#include <vector>
#include <string>

#include <Open3D/Open3D.h>
#include <Cuda/Integration/ScalableTSDFVolumeCuda.h>
#include <Cuda/Integration/ScalableMeshVolumeCuda.h>

#include "examples/Cuda/DatasetConfig.h"

using namespace open3d;

namespace IntegrateScene {
void IntegrateFragment(
    int fragment_id, cuda::ScalableTSDFVolumeCuda<8> &volume,
    DatasetConfig &config) {

    registration::PoseGraph global_pose_graph;
    io::ReadPoseGraph(config.GetPoseGraphFileForRefinedScene(true),
                      global_pose_graph);

    registration::PoseGraph local_pose_graph;
    io::ReadPoseGraph(config.GetPoseGraphFileForFragment(fragment_id, true),
                      local_pose_graph);

    cuda::PinholeCameraIntrinsicCuda intrinsics(config.intrinsic_);
    cuda::RGBDImageCuda rgbd((float) config.max_depth_,
                             (float) config.depth_factor_);

    const int begin = fragment_id * config.n_frames_per_fragment_;
    const int end = std::min((fragment_id + 1) * config.n_frames_per_fragment_,
                             (int) config.color_files_.size());

    for (int i = begin; i < end; ++i) {
        utility::PrintDebug("Integrating frame %d ...\n", i);

        geometry::Image depth, color;
        io::ReadImage(config.depth_files_[i], depth);
        io::ReadImage(config.color_files_[i], color);
        rgbd.Upload(depth, color);

        /* Use ground truth trajectory */
        Eigen::Matrix4d pose = global_pose_graph.nodes_[fragment_id].pose_
            * local_pose_graph.nodes_[i - begin].pose_;
        cuda::TransformCuda trans;
        trans.FromEigen(pose);

        volume.Integrate(rgbd, intrinsics, trans);
    }
}

int Run(DatasetConfig &config) {
    SetVerbosityLevel(utility::VerbosityLevel::VerboseDebug);

    utility::Timer timer;
    timer.Start();

    cuda::TransformCuda trans = cuda::TransformCuda::Identity();
    cuda::ScalableTSDFVolumeCuda<8> tsdf_volume(
        40000, 600000,
        (float) config.tsdf_cubic_size_ / 512,
        (float) config.tsdf_truncation_, trans);

    bool is_success = config.GetFragmentFiles();
    if (!is_success) {
        utility::PrintError("Unable to get fragment files\n");
        return -1;
    }
    for (int i = 0; i < config.fragment_files_.size(); ++i) {
        IntegrateFragment(i, tsdf_volume, config);
    }

    tsdf_volume.GetAllSubvolumes();
    cuda::ScalableMeshVolumeCuda<8> mesher(
        tsdf_volume.active_subvolume_entry_array().size(),
        cuda::VertexWithNormalAndColor, 20000000, 40000000);
    mesher.MarchingCubes(tsdf_volume);
    auto mesh = mesher.mesh().Download();

    io::WriteTriangleMeshToPLY(config.GetReconstructedSceneFile(), *mesh);
    timer.Stop();
    utility::PrintInfo("IntegrateScene takes %.3f s\n", timer.GetDuration() /
        1000.0f);

    return 0;
}
}