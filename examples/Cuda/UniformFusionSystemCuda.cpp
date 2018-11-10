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

#include <Cuda/Integration/UniformTSDFVolumeCuda.h>
#include <opencv2/opencv.hpp>
#include <Cuda/Integration/UniformMeshVolumeCuda.h>

int main(int argc, char *argv[]) {
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

    FPSTimer timer("Process RGBD stream", (int)camera_trajectory->parameters_.size());

    PinholeCameraIntrinsicCuda intrinsics(
        PinholeCameraIntrinsicParameters::PrimeSenseDefault);

    float voxel_length = 0.01f;
    TransformCuda extrinsics = TransformCuda::Identity();
    extrinsics.SetTranslation(Vector3f(-voxel_length * 256));
    UniformTSDFVolumeCuda<512> tsdf_volume(
        voxel_length, 3 * voxel_length, extrinsics);
    UniformMeshVolumeCuda<512> mesher(
        VertexWithNormalAndColor, 4000000, 8000000);

    Image depth, color;
    RGBDImageCuda rgbd(0.1f, 4.0f, 1000.0f);

    VisualizerWithCustomAnimation visualizer;
    if (! visualizer.CreateVisualizerWindow("UniformFusion", 640, 480, 0, 0)) {
        PrintWarning("Failed creating OpenGL window.\n");
        return 0;
    }
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
            ReadImage(dir_name + st[0], depth);
            ReadImage(dir_name + st[1], color);
            rgbd.Upload(depth, color);

            extrinsics.FromEigen(
                camera_trajectory->parameters_[index].extrinsic_.inverse());
            tsdf_volume.Integrate(rgbd, intrinsics, extrinsics);

            mesher.MarchingCubes(tsdf_volume);

            *mesh = mesher.mesh();
            visualizer.PollEvents();
            visualizer.UpdateGeometry();
            visualizer.GetViewControl().ConvertFromPinholeCameraParameters(
                camera_trajectory->parameters_[index]);

            index++;

            if (index == (int)camera_trajectory->parameters_.size()) {
                mesher.MarchingCubes(tsdf_volume);
                WriteTriangleMeshToPLY("uniform-system.ply", *mesher.mesh().Download());
                save_index++;
            }
            timer.Signal();
        }
    }
    fclose(file);
    return 0;
}
