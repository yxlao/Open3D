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
using namespace open3d::utility;
using namespace open3d::io;
using namespace open3d::camera;
using namespace open3d::geometry;
using namespace open3d::visualization;

int main(int argc, char *argv[]) {
    SetVerbosityLevel(VerbosityLevel::VerboseDebug);

    std::string base_path = "/home/wei/Work/data/tum/rgbd_dataset_freiburg3_long_office_household/";
    auto camera_trajectory = CreatePinholeCameraTrajectoryFromFile(base_path + "/trajectory.log");
    auto rgbd_filenames = ReadDataAssociation(base_path + "/data_association.txt");

    int index = 0;
    int save_index = 0;

    FPSTimer timer("Process RGBD stream",
                   (int) camera_trajectory->parameters_.size());

    cuda::PinholeCameraIntrinsicCuda intrinsics(
        PinholeCameraIntrinsicParameters::PrimeSenseDefault);

    float voxel_length = 0.01f;
    cuda::TransformCuda extrinsics = cuda::TransformCuda::Identity();
    extrinsics.SetTranslation(cuda::Vector3f(-voxel_length * 256));
    cuda::UniformTSDFVolumeCuda<512> tsdf_volume(
        voxel_length, 3 * voxel_length, extrinsics);
    cuda::UniformMeshVolumeCuda<512> mesher(
        cuda::VertexWithNormalAndColor, 4000000, 8000000);

    Image depth, color;
    cuda::RGBDImageCuda rgbd(640, 480, 4.0f, 5000.0f);

    VisualizerWithCudaModule visualizer;
    if (! visualizer.CreateVisualizerWindow("UniformFusion", 640, 480, 0, 0)) {
        PrintWarning("Failed creating OpenGL window.\n");
        return 0;
    }
    visualizer.BuildUtilities();
    visualizer.UpdateWindowTitle();

    std::shared_ptr<cuda::TriangleMeshCuda> mesh =
        std::make_shared<cuda::TriangleMeshCuda>();
    visualizer.AddGeometry(mesh);

    for (int i = 0; i < rgbd_filenames.size() - 1; ++i) {
        PrintDebug("Processing frame %d ...\n", index);
        ReadImage(base_path + rgbd_filenames[i].first, depth);
        ReadImage(base_path + rgbd_filenames[i].second, color);
        rgbd.Upload(depth, color);

        /* Use ground truth trajectory */
        Eigen::Matrix4d extrinsic =
            camera_trajectory->parameters_[index].extrinsic_.inverse();

        extrinsics.FromEigen(extrinsic);
        tsdf_volume.Integrate(rgbd, intrinsics, extrinsics);

        mesher.MarchingCubes(tsdf_volume);

        *mesh = mesher.mesh();
        visualizer.PollEvents();
        visualizer.UpdateGeometry();
        visualizer.GetViewControl().ConvertFromPinholeCameraParameters(
            camera_trajectory->parameters_[index]);
        index++;
    }

    return 0;
}
