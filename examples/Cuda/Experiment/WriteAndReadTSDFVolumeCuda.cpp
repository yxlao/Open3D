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

void IntegrateAndWriteFragment(int fragment_id, DatasetConfig &config) {

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
    std::string filename = config.GetBinFileForFragment(fragment_id);
    io::WriteTSDFVolumeToBIN("target-high.bin", tsdf_volume, false);
    timer.Stop();
    utility::PrintInfo("Write TSDF takes %f ms\n", timer.GetDuration());

    cuda::ScalableMeshVolumeCuda mesher(
        cuda::VertexWithNormalAndColor, 8,
        tsdf_volume.active_subvolume_entry_array_.size());
    mesher.MarchingCubes(tsdf_volume);
    auto mesh = mesher.mesh().Download();
    visualization::DrawGeometries({mesh});
//
//    PointCloud pcl;
//    pcl.points_ = mesh->vertices_;
//    pcl.normals_ = mesh->vertex_normals_;
//    pcl.colors_ = mesh->vertex_colors_;
//
//    /** Write original fragments **/
//    timer.Start();
//    WritePointCloudToPLY("test.ply", pcl);
//    timer.Stop();
//    utility::PrintInfo("Write ply takes %f ms\n", timer.GetDuration());
}

void ReadFragment(int fragment_id, DatasetConfig &config) {

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
    io::ReadTSDFVolumeFromBIN("target-high.bin", tsdf_volume, false);
    timer.Stop();
    utility::PrintInfo("Read takes %f ms\n", timer.GetDuration());

    tsdf_volume.GetAllSubvolumes();
    cuda::ScalableMeshVolumeCuda mesher(
        cuda::VertexWithNormalAndColor, 8,
        tsdf_volume.active_subvolume_entry_array_.size());
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
    config.GetFragmentFiles();

    for (int i = 1; i < 2; ++i) {//config.fragment_files_.size(); ++i) {
        utility::PrintInfo("%d\n", i);
        IntegrateAndWriteFragment(i, config);
        ReadFragment(i, config);
    }
}