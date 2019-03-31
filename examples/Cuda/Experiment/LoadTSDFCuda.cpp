//
// Created by wei on 3/31/19.
//

#include <Open3D/Open3D.h>
#include <Cuda/Open3DCuda.h>
#include "../ReconstructionSystem/DatasetConfig.h"

using namespace open3d;
using namespace open3d::registration;
using namespace open3d::geometry;
using namespace open3d::io;
using namespace open3d::utility;

void WriteFragment(int fragment_id, DatasetConfig &config) {

    PoseGraph pose_graph;
    ReadPoseGraph(config.GetPoseGraphFileForFragment(fragment_id, true),
                  pose_graph);

    float voxel_length = config.tsdf_cubic_size_ / 512.0;

    cuda::PinholeCameraIntrinsicCuda intrinsic(config.intrinsic_);
    cuda::TransformCuda trans = cuda::TransformCuda::Identity();
    cuda::ScalableTSDFVolumeCuda<8> tsdf_volume(
        20000,
        400000,
        voxel_length,
        (float) config.tsdf_truncation_,
        trans);

    cuda::RGBDImageCuda rgbd((float) config.max_depth_,
                             (float) config.depth_factor_);

    const int begin = fragment_id * config.n_frames_per_fragment_;
    const int
        end = std::min((fragment_id + 1) * config.n_frames_per_fragment_,
                       (int) config.color_files_.size());

    Timer timer;
    timer.Start();
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
    timer.Stop();
    utility::PrintInfo("Integration takes %f ms\n", timer.GetDuration());

    timer.Start();
    io::WriteTSDFVolumeToBIN("test.bin", tsdf_volume);
    timer.Stop();
    utility::PrintInfo("Write takes %f ms\n", timer.GetDuration());

    cuda::ScalableMeshVolumeCuda<8> mesher(
        tsdf_volume.active_subvolume_entry_array().size(),
        cuda::VertexWithNormalAndColor, 10000000, 20000000);
    mesher.MarchingCubes(tsdf_volume);
    auto mesh = mesher.mesh().Download();
    visualization::DrawGeometries({mesh});
}

void ReadFragment(int fragment_id, DatasetConfig &config) {

    PoseGraph pose_graph;
    ReadPoseGraph(config.GetPoseGraphFileForFragment(fragment_id, true),
                  pose_graph);

    float voxel_length = config.tsdf_cubic_size_ / 512.0;

    cuda::PinholeCameraIntrinsicCuda intrinsic(config.intrinsic_);
    cuda::TransformCuda trans = cuda::TransformCuda::Identity();
    cuda::ScalableTSDFVolumeCuda<8> tsdf_volume(
        20000,
        400000,
        voxel_length,
        (float) config.tsdf_truncation_,
        trans);

    Timer timer;
    timer.Start();
    io::ReadTSDFVolumeFromBIN("test.bin", tsdf_volume);
    timer.Stop();
    utility::PrintInfo("Read takes %f ms\n", timer.GetDuration());

    tsdf_volume.GetAllSubvolumes();
    cuda::ScalableMeshVolumeCuda<8> mesher(
        tsdf_volume.active_subvolume_entry_array().size(),
        cuda::VertexWithNormalAndColor, 10000000, 20000000);
    mesher.MarchingCubes(tsdf_volume);
    auto mesh = mesher.mesh().Download();
    visualization::DrawGeometries({mesh});
}

int main(int argc, char **argv) {
    DatasetConfig config;
    std::string config_path = argc > 1 ? argv[1] :
                              kDefaultDatasetConfigDir + "/stanford/lounge.json";
    bool is_success = io::ReadIJsonConvertible(config_path, config);
    if (!is_success) return 1;

    WriteFragment(0, config);
    ReadFragment(0, config);
}