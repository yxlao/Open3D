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

#include <Open3D/Open3D.h>
#include <Cuda/Open3DCuda.h>

#include "Utils.h"

using namespace open3d;
using namespace open3d::camera;
using namespace open3d::io;
using namespace open3d::geometry;
using namespace open3d::utility;

int main(int argc, char *argv[]) {
    using namespace open3d;

//    std::string base_path = "/home/wei/Work/data/tum/rgbd_dataset_freiburg3_long_office_household/";
    std::string base_path = "/media/wei/Data/data/cmu/a_floor/";
    auto camera_trajectory =
        CreatePinholeCameraTrajectoryFromFile(base_path + "/trajectory.log");
    auto rgbd_filenames =
        ReadDataAssociation(base_path + "/data_association.txt");

    int index = 0;
    cuda::PinholeCameraIntrinsicCuda intrinsics(1368, 1096,
                                                1359.38 / 2,
                                                1359.38 / 2,
                                                1395.56 / 2,
                                                1105.93 / 2);

    float voxel_length = 0.04f;
    cuda::TransformCuda extrinsics = cuda::TransformCuda::Identity();
    cuda::ScalableTSDFVolumeCuda tsdf_volume(
        8, voxel_length, 3 * voxel_length, 8.0f, extrinsics);

    cuda::RGBDImageCuda rgbd(1368, 1096, 8.0f, 1000.0f);
    cuda::ScalableMeshVolumeCuda
        mesher(cuda::VertexWithNormalAndColor, 8, 80000, 20000000, 40000000);


    visualization::VisualizerWithCudaModule visualizer;
    if (!visualizer.CreateVisualizerWindow("ScalableFusion", 640, 480, 0, 0)) {
        PrintWarning("Failed creating OpenGL window.\n");
        return 0;
    }
    visualizer.BuildUtilities();
    visualizer.UpdateWindowTitle();

    std::shared_ptr<cuda::TriangleMeshCuda>
        mesh = std::make_shared<cuda::TriangleMeshCuda>();
    visualizer.AddGeometry(mesh);

    Timer timer;
    for (int i = 0; i < rgbd_filenames.size(); ++i) {
        PrintInfo("Processing frame %d ...\n", index);
        if (i == 112) continue;
        cv::Mat depth = cv::imread(base_path + rgbd_filenames[i].first,
                                   cv::IMREAD_UNCHANGED);
        cv::Mat color = cv::imread(base_path + rgbd_filenames[i].second);
        cv::resize(color, color, cv::Size(0.5 * color.cols, 0.5 * color.rows));
        rgbd.Upload(depth, color);

        /* Use ground truth trajectory */
        Eigen::Matrix4d extrinsic =
            camera_trajectory->parameters_[index].extrinsic_.inverse();

        extrinsics.FromEigen(extrinsic);
        tsdf_volume.Integrate(rgbd, intrinsics, extrinsics);

        timer.Start();
        mesher.MarchingCubes(tsdf_volume);
        timer.Stop();

        *mesh = mesher.mesh();
        visualizer.PollEvents();
        visualizer.UpdateGeometry();
        visualizer.GetViewControl().ConvertFromPinholeCameraParameters(
            camera_trajectory->parameters_[index]);
        index++;
    }

    tsdf_volume.GetAllSubvolumes();
    mesher.MarchingCubes(tsdf_volume);
    auto final_mesh = mesher.mesh().Download();
    WriteTriangleMeshToPLY(base_path + "/integrated.ply", *final_mesh);
}

