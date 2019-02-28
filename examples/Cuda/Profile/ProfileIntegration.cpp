//
// Created by wei on 2/21/19.
//
#include <vector>
#include <string>

#include <Core/Core.h>
#include <IO/IO.h>
#include <Visualization/Visualization.h>

#include <Core/Registration/Registration.h>
#include <Core/Registration/PoseGraph.h>

#include <Cuda/Integration/ScalableTSDFVolumeCuda.h>
#include <Cuda/Integration/ScalableMeshVolumeCuda.h>

#include "examples/Cuda/DatasetConfig.h"
#include "Analyzer.h"

using namespace open3d;

void IntegrateFragmentCuda(
    int fragment_id, cuda::ScalableTSDFVolumeCuda<8> &volume,
    DatasetConfig &config,
    std::vector<double> &times) {

    PoseGraph global_pose_graph;
    ReadPoseGraph(config.GetPoseGraphFileForRefinedScene(true),
                  global_pose_graph);

    PoseGraph local_pose_graph;
    ReadPoseGraph(config.GetPoseGraphFileForFragment(fragment_id, true),
                  local_pose_graph);

    cuda::PinholeCameraIntrinsicCuda intrinsics(config.intrinsic_);
    cuda::RGBDImageCuda rgbd((float) config.max_depth_,
                             (float) config.depth_factor_);

    const int begin = fragment_id * config.n_frames_per_fragment_;
    const int end = std::min((fragment_id + 1) * config.n_frames_per_fragment_,
                             (int) config.color_files_.size());

    Timer timer;
    for (int i = begin; i < end; ++i) {
        Image depth, color;
        ReadImage(config.depth_files_[i], depth);
        ReadImage(config.color_files_[i], color);

        timer.Start();
        rgbd.Upload(depth, color);

        /* Use ground truth trajectory */
        Eigen::Matrix4d pose = global_pose_graph.nodes_[fragment_id].pose_
            * local_pose_graph.nodes_[i - begin].pose_;
        cuda::TransformCuda trans;
        trans.FromEigen(pose);

        volume.Integrate(rgbd, intrinsics, trans);
        timer.Stop();

        double time = timer.GetDuration();
        PrintInfo("Integrate %d takes %f ms\n", i, time);
        times.push_back(time);
    }
}

void IntegrateFragmentCPU(
    int fragment_id, ScalableTSDFVolume &volume,
    DatasetConfig &config,
    std::vector<double> &times) {

    PoseGraph global_pose_graph;
    ReadPoseGraph(config.GetPoseGraphFileForRefinedScene(true),
                  global_pose_graph);

    PoseGraph local_pose_graph;
    ReadPoseGraph(config.GetPoseGraphFileForFragment(fragment_id, true),
                  local_pose_graph);

    const int begin = fragment_id * config.n_frames_per_fragment_;
    const int end = std::min((fragment_id + 1) * config.n_frames_per_fragment_,
                             (int) config.color_files_.size());

    Timer timer;
    for (int i = begin; i < end; ++i) {
        Image depth, color;
        ReadImage(config.depth_files_[i], depth);
        ReadImage(config.color_files_[i], color);

        timer.Start();
        auto rgbd = CreateRGBDImageFromColorAndDepth(color, depth,
                                                     config.depth_factor_,
                                                     config.max_depth_, false);

        /* Use ground truth trajectory */
        Eigen::Matrix4d pose = global_pose_graph.nodes_[fragment_id].pose_
            * local_pose_graph.nodes_[i - begin].pose_;

        /** CPU version receives world to camera **/
        volume.Integrate(*rgbd, config.intrinsic_, pose.inverse());
        timer.Stop();

        double time = timer.GetDuration();
        if (i % 100 == 0) {
            PrintInfo("Integrate %d takes %f ms\n", i, time);
        }
        times.push_back(time);
    }
}

int main(int argc, char **argv) {
    DatasetConfig config;

    std::string config_path = argc > 1 ? argv[1] :
                              "/home/wei/Work/projects/dense_mapping/Open3D/examples/Cuda"
                              "/ReconstructionSystem/config/livingroom1-simulated"
                              ".json";

    bool is_success = ReadIJsonConvertible(config_path, config);
    if (!is_success) return 1;
    is_success = config.GetFragmentFiles();
    if (!is_success) {
        PrintError("Unable to get fragment files\n");
        return -1;
    }

    cuda::TransformCuda trans = cuda::TransformCuda::Identity();
    cuda::ScalableTSDFVolumeCuda<8> tsdf_volume(
        40000, 600000,
        (float) config.tsdf_cubic_size_ / 512,
        (float) config.tsdf_truncation_, trans);
    ScalableTSDFVolume tsdf_volume_cpu(
        config.tsdf_cubic_size_ / 512,
        config.tsdf_truncation_,
        TSDFVolumeColorType::RGB8);

    std::vector<double> times_gpu;
    for (int i = 0; i < 30; ++i) {
        IntegrateFragmentCuda(i, tsdf_volume, config, times_gpu);
    }
    double mean, std;


    tsdf_volume.GetAllSubvolumes();
    cuda::ScalableMeshVolumeCuda<8> mesher(
        tsdf_volume.active_subvolume_entry_array().size(),
        cuda::VertexWithNormalAndColor, 20000000, 40000000);
    mesher.MarchingCubes(tsdf_volume);
    auto mesh = mesher.mesh().Download();
    DrawGeometries({mesh});


//    std::tie(mean, std) = ComputeStatistics(times_gpu);
//    PrintInfo("gpu time: avg = %f, std = %f\n", mean, std);

//    std::vector<double> times_cpu;
//    for (int i = 0; i < config.fragment_files_.size(); ++i) {
//        IntegrateFragmentCPU(i, tsdf_volume_cpu, config, times_cpu);
//    }
//    std::tie(mean, std) = ComputeStatistics(times_cpu);
//    PrintInfo("cpu time: avg = %f, std = %f\n", mean, std);

    return 0;
}