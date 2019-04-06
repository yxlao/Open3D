//
// Created by wei on 4/5/19.
//

#include <Open3D/Open3D.h>
#include <Cuda/Open3DCuda.h>
#include <Cuda/Integration/Experiment/ScalableTSDFVolumeProcessorCuda.h>
#include <Cuda/Integration/Experiment/ScalableVolumeRegistrationCuda.h>
#include <Cuda/Visualization/Visualizer/VisualizerWithCudaModule.h>
#include <Open3D/Visualization/Visualizer/Visualizer.h>
#include "../ReconstructionSystem/DatasetConfig.h"

using namespace open3d;
using namespace open3d::registration;
using namespace open3d::geometry;
using namespace open3d::io;
using namespace open3d::utility;
using namespace open3d::visualization;

cuda::ScalableTSDFVolumeCuda
ReadTSDFVolume(const std::string &filename, DatasetConfig &config) {
    float voxel_length = config.tsdf_cubic_size_ / 512.0;

    cuda::TransformCuda trans = cuda::TransformCuda::Identity();
    cuda::ScalableTSDFVolumeCuda tsdf_volume(
        8, voxel_length * 4, (float) config.tsdf_truncation_ * 4, trans);

    Timer timer;
    timer.Start();

    io::ReadTSDFVolumeFromBIN(filename, tsdf_volume);
    return tsdf_volume;
}

std::shared_ptr<PointCloud> ExtractVoxelsNearSurface(cuda::ScalableTSDFVolumeCuda &volume) {
    cuda::ScalableTSDFVolumeProcessorCuda gradient_volume_target(
        8, volume.active_subvolume_entry_array_.size());
    gradient_volume_target.ComputeGradient(volume);
    cuda::PointCloudCuda pcl_target = gradient_volume_target.ExtractVoxelsNearSurface(
        volume, 0.8f);
    return pcl_target.Download();
}

int RegistrationForTSDFVolumes(
    const std::string &source_path,
    const std::string &target_path,
    const Eigen::Matrix4d &init_source_to_target,
    DatasetConfig &config) {

    SetVerbosityLevel(VerbosityLevel::VerboseDebug);

    auto source = ReadTSDFVolume(source_path, config);
    auto target = ReadTSDFVolume(target_path, config);
    source.GetAllSubvolumes();
    target.GetAllSubvolumes();

    auto pcl_source = ExtractVoxelsNearSurface(source);
    auto pcl_target = ExtractVoxelsNearSurface(target);
    pcl_source->Transform(init_source_to_target);


    cuda::ScalableVolumeRegistrationCuda registration;
    registration.Initialize(source, target, init_source_to_target);

    /** Prepare visualizer **/
    visualization::VisualizerWithCudaModule visualizer;
    if (!visualizer.CreateVisualizerWindow("TSDF-2-TSDF", 640, 480, 0, 0)) {
        PrintWarning("Failed creating OpenGL window.\n");
        return -1;
    }
    visualizer.BuildUtilities();
    visualizer.UpdateWindowTitle();
    visualizer.AddGeometry(pcl_source);
    visualizer.AddGeometry(pcl_target);

    bool finished = false;
    int iter = 0, max_iter = 800;

    Eigen::Matrix4d prev_pose = init_source_to_target;
    visualizer.RegisterKeyCallback(GLFW_KEY_SPACE, [&](Visualizer *vis) {
        if (finished) return false;

        /* Registration (1 iteration) */
        auto delta = registration.DoSingleIteration(iter++);

        Eigen::Matrix4d curr_pose = registration.trans_source_to_target_;
        pcl_source->Transform(curr_pose * prev_pose.inverse());
        prev_pose = curr_pose;

        /* Updated source */
        vis->UpdateGeometry();

        /* Update flags */
        if (iter >= max_iter)
            finished = true;
        return !finished;
    });

    bool should_close = false;
    while (!should_close) {
        should_close = !visualizer.PollEvents();
    }
    visualizer.DestroyVisualizerWindow();

    return 0;
}


int main(int argc, char **argv) {
    DatasetConfig config;
    std::string config_path = argc > 1 ? argv[1] :
                              kDefaultDatasetConfigDir + "/stanford/lounge.json";
    bool is_success = io::ReadIJsonConvertible(config_path, config);
    if (!is_success) return 1;

    PoseGraph pose_graph_s;
    ReadPoseGraph(config.GetPoseGraphFileForFragment(0, true),
                  pose_graph_s);
    auto rbegin = pose_graph_s.nodes_.rbegin();
    Eigen::Matrix4d init_source_to_target = rbegin->pose_.inverse();
    init_source_to_target = pose_graph_s.nodes_[70].pose_.inverse();

//    init_source_to_target = pose_graph_s.nodes_[3].pose_;
//    std::cout << init_source_to_target << std::endl;

    RegistrationForTSDFVolumes("target.bin", "source.bin",
                               init_source_to_target.inverse(), config);
//    RegistrationForTSDFVolumes("source.bin", "target.bin",
//                               init_source_to_target, config);

//    auto source = ReadTSDFVolume("source.bin", config);
//    auto target = ReadTSDFVolume("target.bin", config);
//
//    cuda::ScalableVolumeRegistrationCuda registration;
//    registration.Initialize(source, target, init_source_to_target);
//
//    utility::SetVerbosityLevel(utility::VerbosityLevel::VerboseDebug);
//    for (int i = 0; i < 60; ++i) {
//        registration.DoSingleIteration(i);
//    }
}