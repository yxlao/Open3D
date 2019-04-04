//
// Created by wei on 4/4/19.
//

#include <Open3D/Open3D.h>
#include <Cuda/Open3DCuda.h>
#include <Cuda/Integration/Experiment/ScalableTSDFVolumeProcessorCuda.h>
#include "../ReconstructionSystem/DatasetConfig.h"

using namespace open3d;
using namespace open3d::registration;
using namespace open3d::geometry;
using namespace open3d::io;
using namespace open3d::utility;

void ReadAndComputeGradient(int fragment_id, DatasetConfig &config) {

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

    tsdf_volume.GetAllSubvolumes();
    cuda::ScalableTSDFVolumeProcessorCuda gradient_volume(
        8, tsdf_volume.active_subvolume_entry_array_.size());
    gradient_volume.ComputeGradient(tsdf_volume);

    cuda::PointCloudCuda pcl = gradient_volume.ExtractVoxelsNearSurface(
        tsdf_volume, 0.8f);
    auto pcl_cpu = pcl.Download();

    visualization::DrawGeometries({pcl_cpu});
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
        ReadAndComputeGradient(i, config);
    }
}