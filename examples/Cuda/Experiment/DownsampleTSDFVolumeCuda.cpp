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

void IntegrateForMultiResSubvolume(int fragment_id,
                                   DatasetConfig &config) {

    PoseGraph pose_graph;
    ReadPoseGraph(config.GetPoseGraphFileForFragment(fragment_id, true),
                  pose_graph);

    float voxel_length = config.tsdf_cubic_size_ / 512.0;

    cuda::PinholeCameraIntrinsicCuda intrinsic(config.intrinsic_);
    cuda::TransformCuda trans = cuda::TransformCuda::Identity();

    cuda::ScalableTSDFVolumeCuda tsdf_volume(
        8, voxel_length, (float) config.tsdf_truncation_, trans),
    tsdf_volume_2(
        8, voxel_length * 2, (float) config.tsdf_truncation_ * 2, trans,
        tsdf_volume.bucket_count_, tsdf_volume.value_capacity_ / 2),
    tsdf_volume_4(
        8, voxel_length * 4, (float) config.tsdf_truncation_ * 4, trans,
        tsdf_volume.bucket_count_, tsdf_volume.value_capacity_ / 4);

    cuda::RGBDImageCuda rgbd((float) config.max_depth_,
                             (float) config.depth_factor_);
    const int begin = fragment_id * config.n_frames_per_fragment_;
    const int end = std::min((fragment_id + 1) * config.n_frames_per_fragment_,
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
        tsdf_volume_2.Integrate(rgbd, intrinsic, trans);
        tsdf_volume_4.Integrate(rgbd, intrinsic, trans);
    }

    tsdf_volume.GetAllSubvolumes();
    utility::PrintInfo("Total subvolumes: %d\n", tsdf_volume.active_subvolume_entry_array_.size());
    tsdf_volume_2.GetAllSubvolumes();
    utility::PrintInfo("Total subvolumes: %d\n", tsdf_volume_2.active_subvolume_entry_array_.size());
    tsdf_volume_4.GetAllSubvolumes();
    utility::PrintInfo("Total subvolumes: %d\n", tsdf_volume_4.active_subvolume_entry_array_.size());

    timer.Stop();
    utility::PrintInfo("Integration takes %f ms\n", timer.GetDuration());

    cuda::ScalableMeshVolumeCuda mesher(
        cuda::VertexWithNormalAndColor, 8,
        tsdf_volume_4.active_subvolume_entry_array_.size());
    mesher.MarchingCubes(tsdf_volume_4);
    auto mesh = mesher.mesh().Download();
    visualization::DrawGeometries({mesh});
}


void IntegrateForOriginResolution(int fragment_id,
                                  DatasetConfig &config) {

    PoseGraph pose_graph;
    ReadPoseGraph(config.GetPoseGraphFileForFragment(fragment_id, true),
                  pose_graph);

    float voxel_length = config.tsdf_cubic_size_ / 512.0;

    cuda::PinholeCameraIntrinsicCuda intrinsic(config.intrinsic_);
    cuda::TransformCuda trans = cuda::TransformCuda::Identity();

    cuda::ScalableTSDFVolumeCuda tsdf_volume(
        8, voxel_length, (float) config.tsdf_truncation_, trans);

    cuda::RGBDImageCuda rgbd((float) config.max_depth_,
                             (float) config.depth_factor_);

    const int begin = fragment_id * config.n_frames_per_fragment_;
    const int end = std::min((fragment_id + 1) * config.n_frames_per_fragment_,
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
    utility::PrintInfo("Total subvolumes: %d\n", tsdf_volume.active_subvolume_entry_array_.size());


    timer.Stop();
    utility::PrintInfo("Integration takes %f ms\n", timer.GetDuration());

    cuda::ScalableMeshVolumeCuda mesher(
        cuda::VertexWithNormalAndColor, 8,
        tsdf_volume.active_subvolume_entry_array_.size());
    mesher.MarchingCubes(tsdf_volume);
    auto mesh = mesher.mesh().Download();
    visualization::DrawGeometries({mesh});
}


void IntegrateForCoarseSubvolume(int fragment_id,
                                 DatasetConfig &config,
                                 int scale) {

    PoseGraph pose_graph;
    ReadPoseGraph(config.GetPoseGraphFileForFragment(fragment_id, true),
                  pose_graph);

    float voxel_length = config.tsdf_cubic_size_ / 512.0;

    cuda::PinholeCameraIntrinsicCuda intrinsic(config.intrinsic_);
    cuda::TransformCuda trans = cuda::TransformCuda::Identity();

    int factor = 1 << (scale - 1);
    cuda::ScalableTSDFVolumeCuda tsdf_volume(
        8, voxel_length * factor, (float) config.tsdf_truncation_ * factor,
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
    utility::PrintInfo("Total subvolumes: %d\n", tsdf_volume.active_subvolume_entry_array_.size());

    timer.Stop();
    utility::PrintInfo("Integration takes %f ms\n", timer.GetDuration());

    cuda::ScalableMeshVolumeCuda mesher(
        cuda::VertexWithNormalAndColor, 8,
        tsdf_volume.active_subvolume_entry_array_.size());
    mesher.MarchingCubes(tsdf_volume);
    auto mesh = mesher.mesh().Download();
    visualization::DrawGeometries({mesh});
}

void ReadAndDownsampleFragment(int fragment_id, DatasetConfig &config) {

    PoseGraph pose_graph;
    ReadPoseGraph(config.GetPoseGraphFileForFragment(fragment_id, true),
                  pose_graph);

    float voxel_length = config.tsdf_cubic_size_ / 512.0;

    cuda::TransformCuda trans = cuda::TransformCuda::Identity();
    cuda::ScalableTSDFVolumeCuda tsdf_volume(
        8, voxel_length, (float) config.tsdf_truncation_, trans);

    Timer timer;
    timer.Start();

    std::string filename = config.GetBinFileForFragment(fragment_id);
    io::ReadTSDFVolumeFromBIN(filename, tsdf_volume);
    timer.Stop();
    utility::PrintInfo("Read takes %f ms\n", timer.GetDuration());

    timer.Start();
    auto tsdf_volume_down_2 = tsdf_volume.DownSample();
    auto tsdf_volume_down_4 = tsdf_volume_down_2.DownSample();
    timer.Stop();
    utility::PrintInfo("Downsample takes %f ms\n", timer.GetDuration());

    tsdf_volume_down_4.GetAllSubvolumes();
    utility::PrintInfo("tsdf_volume_down.active: %d\n",
        tsdf_volume_down_4.active_subvolume_entry_array_.size());

    cuda::ScalableMeshVolumeCuda mesher(
        cuda::VertexWithNormalAndColor, 2,
        tsdf_volume_down_4.active_subvolume_entry_array_.size());
    mesher.MarchingCubes(tsdf_volume_down_4);
    auto mesh = mesher.mesh().Download();
    visualization::DrawGeometries({mesh});
}

int main(int argc, char **argv) {
    DatasetConfig config;
    std::string config_path = argc > 1 ? argv[1] :
                              kDefaultDatasetConfigDir + "/stanford/lounge.json";
    bool is_success = io::ReadIJsonConvertible(config_path, config);
    if (!is_success) return 1;
    config.GetFragmentFiles();

    for (int i = 0; i < config.fragment_files_.size(); ++i) {
        utility::PrintInfo("%d\n", i);
        IntegrateForMultiResSubvolume(i, config);
        IntegrateForOriginResolution(i, config);
        IntegrateForCoarseSubvolume(i, config, 3);
        ReadAndDownsampleFragment(i, config);
    }
}