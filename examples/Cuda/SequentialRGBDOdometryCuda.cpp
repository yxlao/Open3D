//
// Created by wei on 10/6/18.
//

#include <string>
#include <vector>
#include <Core/Core.h>
#include <IO/IO.h>
#include <Cuda/Odometry/RGBDOdometryCuda.h>
#include <Cuda/Integration/ScalableTSDFVolumeCuda.h>
#include <Cuda/Integration/ScalableMeshVolumeCuda.h>
#include <Visualization/Visualization.h>

#include "ReadDataAssociation.h"

using namespace open3d;

void PrintHelp() {
    PrintOpen3DVersion();
    PrintInfo("Usage :\n");
    PrintInfo("    > SequentialRGBDOdometryCuda [dataset_path]\n");
}

int main(int argc, char **argv) {
    if (argc != 2 || ProgramOptionExists(argc, argv, "--help")) {
        PrintHelp();
        return 1;
    }

    SetVerbosityLevel(VerbosityLevel::VerboseDebug);

    std::string base_path = argv[1];
    auto rgbd_filenames = ReadDataAssociation(
        base_path + "/data_association.txt");

    int index = 0;
    int save_index = 0;

    FPSTimer timer("Process RGBD stream", rgbd_filenames.size());

    PinholeCameraIntrinsicCuda intrinsics(
        PinholeCameraIntrinsicParameters::PrimeSenseDefault);

    float voxel_length = 0.01f;
    TransformCuda extrinsics = TransformCuda::Identity();
    ScalableTSDFVolumeCuda<8> tsdf_volume(
        10000, 200000, voxel_length, 3 * voxel_length, extrinsics);

    Image depth, color;
    RGBDImageCuda rgbd_prev(0.1f, 4.0f, 1000.0f);
    RGBDImageCuda rgbd_curr(0.1f, 4.0f, 1000.0f);
    ScalableMeshVolumeCuda<8> mesher(
        40000, VertexWithNormalAndColor, 6000000, 12000000);

    RGBDOdometryCuda<3> odometry;
    odometry.SetIntrinsics(PinholeCameraIntrinsic(
        PinholeCameraIntrinsicParameters::PrimeSenseDefault));
    odometry.SetParameters(OdometryOption(), 0.5f);

    VisualizerWithCustomAnimation visualizer;
    if (!visualizer.CreateVisualizerWindow("SequentialRGBDOdometryCuda", 640, 480, 0, 0)) {
        PrintWarning("Failed creating OpenGL window.\n");
        return -1;
    }
    visualizer.BuildUtilities();
    visualizer.UpdateWindowTitle();

    std::shared_ptr<TriangleMeshCuda>
        mesh = std::make_shared<TriangleMeshCuda>();
    visualizer.AddGeometry(mesh);

    Eigen::Matrix4d target_to_world = Eigen::Matrix4d::Identity();

    PinholeCameraParameters params;
    params.intrinsic_ = PinholeCameraIntrinsicParameters::PrimeSenseDefault;

    for (int i = 0; i < rgbd_filenames.size() - 1; ++i) {
        ReadImage(base_path + "/" + rgbd_filenames[i].first, depth);
        ReadImage(base_path + "/" + rgbd_filenames[i].second, color);
        rgbd_curr.Upload(depth, color);

        if (index >= 1) {
            odometry.transform_source_to_target_ =
                Eigen::Matrix4d::Identity();
            odometry.Initialize(rgbd_curr, rgbd_prev);
            odometry.ComputeMultiScale();
            target_to_world =
                target_to_world * odometry.transform_source_to_target_;
        }
        std::cout << index << std::endl;

        extrinsics.FromEigen(target_to_world);
        tsdf_volume.Integrate(rgbd_curr, intrinsics, extrinsics);

        mesher.MarchingCubes(tsdf_volume);

        *mesh = mesher.mesh();
        visualizer.PollEvents();
        visualizer.UpdateGeometry();

        params.extrinsic_ = extrinsics.ToEigen().inverse();
        std::cout << params.extrinsic_ << std::endl;
        visualizer.GetViewControl().ConvertFromPinholeCameraParameters(params);
        index++;

        if (index > 0 && index % 200 == 0) {
            tsdf_volume.GetAllSubvolumes();
            mesher.MarchingCubes(tsdf_volume);
            WriteTriangleMeshToPLY(
                "fragment-" + std::to_string(save_index) + ".ply",
                *mesher.mesh().Download());
            save_index++;
        }

        rgbd_prev.CopyFrom(rgbd_curr);
        timer.Signal();
    }

    return 0;
}