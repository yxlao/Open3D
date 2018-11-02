//
// Created by wei on 10/31/18.
//

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

#include <Core/Core.h>
#include <IO/IO.h>
#include <Visualization/Visualization.h>
#include <opencv2/opencv.hpp>
#include <Geometry/ImageCuda.h>
#include <Geometry/PinholeCameraCuda.h>
#include <Geometry/TransformCuda.h>
#include <Geometry/TriangleMeshCuda.h>
#include <Integration/ScalableTSDFVolumeCuda.h>
#include <Integration/ScalableMeshVolumeCuda.h>


int main(int argc, char **argv)
{
    using namespace open3d;
    using namespace open3d::filesystem;

    using namespace open3d;
    cv::Mat im = cv::imread("../../examples/TestData/RGBD/depth/apt-022640.png",
                            cv::IMREAD_UNCHANGED);
    ImageCuda<Vector1s> imcuda;
    imcuda.Upload(im);
    auto imcudaf = imcuda.ToFloat(0.001f);

    MonoPinholeCameraCuda intrinsics;
    intrinsics.SetUp();

    float voxel_length = 0.01f;
    TransformCuda extrinsics = TransformCuda::Identity();
    ScalableTSDFVolumeCuda<8> tsdf_volume(10000, 200000,
                                          voxel_length, 3 * voxel_length,
                                          extrinsics);
    Timer timer;
    timer.Start();
    for (int i = 0; i < 10; ++i) {
        tsdf_volume.Integrate(imcudaf, intrinsics, extrinsics);
    }
    timer.Stop();
    PrintInfo("Integration takes: %f milliseconds\n", timer.GetDuration() / 10);

    ScalableMeshVolumeCuda<8> mesher(10000, VertexWithNormal, 100000, 200000);
    mesher.active_subvolumes_ = tsdf_volume.active_subvolume_entry_array().size();

    PrintInfo("Active subvolumes: %d\n", mesher.active_subvolumes_);

    timer.Start();
    int iter = 100;
    for (int i = 0; i < iter; ++i) {
        mesher.MarchingCubes(tsdf_volume);
    }
    timer.Stop();
    PrintInfo("MarchingCubes takes: %f milliseconds\n", timer.GetDuration() / iter);

    std::shared_ptr<TriangleMesh> mesh = mesher.mesh().Download();
    mesh->ComputeVertexNormals();

    std::vector<std::shared_ptr<Geometry>> geometry_ptrs;

    VisualizerWithCustomAnimation visualizer;
    if (! visualizer.CreateVisualizerWindow("test", 640, 480, 0, 0)) {
        PrintWarning("Failed creating OpenGL window.\n");
        return 0;
    }
    visualizer.AddGeometry(mesh);

    if (visualizer.HasGeometry() == false) {
        PrintWarning("No geometry to render!\n");
        visualizer.DestroyVisualizerWindow();
        return 0;
    }

    visualizer.GetRenderOption().show_coordinate_frame_ = true;
    visualizer.Run();
    visualizer.DestroyVisualizerWindow();

    return 1;
}
