// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// The MIT License (MIT)
//
// Copyright (c) 2018 www.open3d.org
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
// FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
// IN THE SOFTWARE.
// ----------------------------------------------------------------------------

#include <iostream>
#include <memory>

#include <Core/Core.h>
#include <IO/IO.h>
#include <Visualization/Visualization.h>

#include <Cuda/Integration/ScalableTSDFVolumeCuda.h>
#include <opencv2/opencv.hpp>
#include <Cuda/Integration/ScalableMeshVolumeCuda.h>

int main(int argc, char *argv[])
{
    using namespace open3d;

    std::string match_filename = "/home/wei/Work/data/lounge/data_association.txt";
    std::string log_filename = "/home/wei/Work/data/lounge/lounge_trajectory.log";

    auto camera_trajectory = CreatePinholeCameraTrajectoryFromFile(
        log_filename);
    std::string dir_name = filesystem::GetFileParentDirectory(
        match_filename).c_str();
    FILE *file = fopen(match_filename.c_str(), "r");
    if (file == NULL) {
        PrintError("Unable to open file %s\n", match_filename.c_str());
        fclose(file);
        return 0;
    }
    char buffer[DEFAULT_IO_BUFFER_SIZE];
    int index = 0;
    int save_index = 0;

    PinholeCameraIntrinsicCuda intrinsics(
        PinholeCameraIntrinsicParameters::PrimeSenseDefault);

    float voxel_length = 0.01f;
    TransformCuda extrinsics = TransformCuda::Identity();
    ScalableTSDFVolumeCuda<8> tsdf_volume(10000, 200000,
                                          voxel_length, 3 * voxel_length,
                                          extrinsics);

    FPSTimer timer("Process RGBD stream", (int)camera_trajectory->extrinsic_.size());

    RGBDImageCuda rgbd(0.1f, 4.0f, 1000.0f);

    SetVerbosityLevel(VerbosityLevel::VerboseDebug);
    ScalableMeshVolumeCuda<8> mesher(40000,
                                     VertexWithNormalAndColor,
                                     6000000, 12000000);

    VisualizerWithCustomAnimation visualizer;
    if (! visualizer.CreateVisualizerWindow("test", 640, 480, 0, 0)) {
        PrintWarning("Failed creating OpenGL window.\n");
        return 0;
    }
    visualizer.GetRenderOption().show_coordinate_frame_ = true;
    //visualizer.GetRenderOption().mesh_color_option_
    //=RenderOption::MeshColorOption::Normal;
    visualizer.GetRenderOption().mesh_show_back_face_ = true;
    visualizer.BuildUtilities();
    visualizer.UpdateWindowTitle();

    std::shared_ptr<TriangleMeshCuda> mesh = std::make_shared<TriangleMeshCuda>();
    visualizer.AddGeometry(mesh);

    while (fgets(buffer, DEFAULT_IO_BUFFER_SIZE, file)) {
        std::vector<std::string> st;
        SplitString(st, buffer, "\t\r\n ");
        if (st.size() >= 2) {
            PrintDebug("Processing frame %d ...\n", index);
            cv::Mat depth = cv::imread(dir_name + st[0], cv::IMREAD_UNCHANGED);
            cv::Mat color = cv::imread(dir_name + st[1]);
            cv::cvtColor(color, color, cv::COLOR_BGR2RGB);

            rgbd.Upload(depth, color);

            Eigen::Matrix4d extrinsic = camera_trajectory->extrinsic_[index].inverse();

            extrinsics.FromEigen(extrinsic);
            tsdf_volume.Integrate(rgbd, intrinsics, extrinsics);

            mesher.MarchingCubes(tsdf_volume);

            *mesh = mesher.mesh();
            visualizer.PollEvents();
            visualizer.UpdateGeometry();
            visualizer.GetViewControl().ConvertFromPinholeCameraParameters(
                camera_trajectory->intrinsic_,
                camera_trajectory->extrinsic_[index]);

            index++;

            if (index == (int)camera_trajectory->extrinsic_.size()) {
                tsdf_volume.GetAllSubvolumes();
                mesher.MarchingCubes(tsdf_volume);
                WriteTriangleMeshToPLY("system.ply", *mesher.mesh().Download());
                save_index++;
            }
            timer.Signal();
        }
    }
    fclose(file);
    return 0;
}
