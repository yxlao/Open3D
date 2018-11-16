//
// Created by wei on 10/6/18.
//

#include <string>
#include <vector>
#include <Core/Core.h>
#include <IO/IO.h>
#include <Cuda/Odometry/SequentialRGBDOdometryCuda.h>
#include <Cuda/Integration/ScalableTSDFVolumeCuda.h>
#include <Cuda/Integration/ScalableMeshVolumeCuda.h>
#include <Visualization/Visualization.h>

#include <opencv2/opencv.hpp>

void f() {
    using namespace open3d;
    SetVerbosityLevel(VerbosityLevel::VerboseDebug);

    std::string
        match_filename = "/home/wei/Work/data/apartment/data_association.txt";
    std::string
        log_filename = "/home/wei/Work/data/apratment/apartment.log";

    auto camera_trajectory = CreatePinholeCameraTrajectoryFromFile(log_filename);
    std::string dir_name = filesystem::GetFileParentDirectory(match_filename).c_str();
    FILE *file = fopen(match_filename.c_str(), "r");
    if (file == NULL) {
        PrintError("Unable to open file %s\n", match_filename.c_str());
        fclose(file);
        return;
    }
    char buffer[DEFAULT_IO_BUFFER_SIZE];
    int index = 0;
    int save_index = 0;

    FPSTimer timer("Process RGBD stream",
                   (int) camera_trajectory->parameters_.size());

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
    odometry.SetParameters(0.0f, 0.1f, 4.0f, 0.07f);

    VisualizerWithCustomAnimation visualizer;
    if (!visualizer.CreateVisualizerWindow("ScalableFusion", 640, 480, 0, 0)) {
        PrintWarning("Failed creating OpenGL window.\n");
        return;
    }
    visualizer.BuildUtilities();
    visualizer.UpdateWindowTitle();

    std::shared_ptr<TriangleMeshCuda>
        mesh = std::make_shared<TriangleMeshCuda>();
    visualizer.AddGeometry(mesh);

    Eigen::Matrix4d target_to_world = Eigen::Matrix4d::Identity();

    PinholeCameraParameters params;
    params.intrinsic_ = PinholeCameraIntrinsicParameters::PrimeSenseDefault;

    while (fgets(buffer, DEFAULT_IO_BUFFER_SIZE, file)) {
        std::vector<std::string> st;
        SplitString(st, buffer, "\t\r\n ");
        if (st.size() >= 2) {
            PrintDebug("Processing frame %d ...\n", index);
            ReadImage(dir_name + st[0], depth);
            ReadImage(dir_name + st[1], color);
            rgbd_curr.Upload(depth, color);

            if (index >= 1) {
                odometry.transform_source_to_target_ =
                    Eigen::Matrix4d::Identity();
                odometry.PrepareData(rgbd_curr, rgbd_prev);
                odometry.Apply();
                target_to_world = target_to_world * odometry.transform_source_to_target_;
            }

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
                WriteTriangleMeshToPLY("fragment-" + std::to_string(save_index) + ".ply",
                                       *mesher.mesh().Download());
                save_index++;
            }

            rgbd_prev.CopyFrom(rgbd_curr);
            timer.Signal();
        }
    }

    fclose(file);
}

int main(int argc, char**argv) {
    using namespace open3d;

    f();
    SetVerbosityLevel(VerbosityLevel::VerboseDebug);
//
//    std::string base_path = // "../../../examples/TestData/RGBD/";
//        "/home/wei/Work/data/lounge/";
//
//    Image source_color, source_depth, target_color, target_depth;
//    ReadImage(base_path + "image/000002.png", source_color);
//    ReadImage(base_path + "depth/000002.png", source_depth);
//
//    ReadImage(base_path + "image/000001.png", target_color);
//    ReadImage(base_path + "depth/000001.png", target_depth);
//
//    RGBDImageCuda source, target;
//    source.Upload(source_depth, source_color);
//    target.Upload(target_depth, target_color);
//
//    RGBDOdometryCuda<3> odometry;
//    odometry.SetIntrinsics(PinholeCameraIntrinsic(
//        PinholeCameraIntrinsicParameters::PrimeSenseDefault));
//
//    odometry.SetParameters(0.2f, 0.1f, 4.0f, 0.07f);
//
//    Timer timer;
//    const int num_iters = 1;
//
//    timer.Start();
//    odometry.PrepareData(source, target);
//    for (int i = 0; i < num_iters; ++i) {
//        odometry.transform_source_to_target_ = Eigen::Matrix4d::Identity();
//        odometry.Apply();
//    }
//    timer.Stop();
//    PrintInfo("Average odometry time: %f milliseconds.\n",
//              timer.GetDuration() / num_iters);
}