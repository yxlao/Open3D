//
// Created by wei on 4/3/19.
//

#include <Open3D/Open3D.h>
#include <Cuda/Open3DCuda.h>
#include "../ReconstructionSystem/DatasetConfig.h"

using namespace open3d;
using namespace open3d::registration;
using namespace open3d::geometry;
using namespace open3d::io;
using namespace open3d::utility;

void ReadAndRayCasting(int fragment_id, DatasetConfig &config) {

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

    cuda::PinholeCameraIntrinsicCuda intrinsic(config.intrinsic_);
    for (int i = 0; i < pose_graph.nodes_.size(); ++i) {
        Eigen::Matrix4d pose = pose_graph.nodes_[i].pose_;
        trans.FromEigen(pose);
        cuda::ImageCuda<float, 3> raycaster(640, 480);
        tsdf_volume.RayCasting(raycaster, intrinsic, trans);
        cv::Mat im = raycaster.DownloadMat();
        cv::imshow("raycasting", im);
        cv::waitKey(-1);
    }
}

void ReadAndVolumeRendering(int fragment_id, DatasetConfig &config, cv::VideoWriter &writer) {

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

    cuda::PinholeCameraIntrinsicCuda intrinsic(config.intrinsic_);
    for (int i = 0; i < pose_graph.nodes_.size(); ++i) {
        Eigen::Matrix4d pose = pose_graph.nodes_[i].pose_;
        trans.FromEigen(pose);
        cuda::ImageCuda<float, 3> raycaster(640, 480);
        tsdf_volume.VolumeRendering(raycaster, intrinsic, trans);
        cv::Mat im = raycaster.DownloadMat();
        cv::imshow("raycasting", im);
        cv::waitKey(10);

        im.convertTo(im, CV_8UC3, 255);
        writer << im;
    }
}

int main(int argc, char **argv) {
    DatasetConfig config;
    std::string config_path = argc > 1 ? argv[1] :
                              kDefaultDatasetConfigDir + "/stanford/lounge.json";
    bool is_success = io::ReadIJsonConvertible(config_path, config);
    if (!is_success) return 1;
    config.GetFragmentFiles();

    cv::VideoWriter writer("volume_rendering.mp4", 0x21 /* H264 */, 24,
                           cv::Size(640, 480));


    for (int i = 0; i < config.fragment_files_.size(); ++i) {
        utility::PrintInfo("%d\n", i);
        ReadAndVolumeRendering(i, config, writer);
    }
}