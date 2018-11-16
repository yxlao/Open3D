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
#include <Cuda/Camera/PinholeCameraIntrinsicCuda.h>
#include <Cuda/Common/TransformCuda.h>
#include <Geometry/TriangleMeshCuda.h>
#include <Geometry/RGBDImageCuda.h>
#include <Integration/ScalableTSDFVolumeCuda.h>
#include <Integration/ScalableMeshVolumeCuda.h>

int main(int argc, char **argv)
{
    using namespace open3d;
    using namespace open3d::filesystem;

    Image depth, color;
    ReadImage("../../../examples/TestData/RGBD/depth/00000.png", depth);
    ReadImage("../../../examples/TestData/RGBD/color/00000.jpg", color);

    RGBDImageCuda rgbd(0.1f, 3.5f, 1000.0f);
    rgbd.Upload(depth, color);

    PinholeCameraIntrinsicCuda intrinsics(
        PinholeCameraIntrinsicParameters::PrimeSenseDefault);

    float voxel_length = 0.01f;
    TransformCuda extrinsics = TransformCuda::Identity();
    ScalableTSDFVolumeCuda<8> tsdf_volume(
        10000, 200000, voxel_length, 3 * voxel_length, extrinsics);
    tsdf_volume.Integrate(rgbd, intrinsics, extrinsics);
    ScalableMeshVolumeCuda<8> mesher(
        10000, VertexWithNormalAndColor, 100000, 200000);
    mesher.active_subvolumes_ = tsdf_volume.active_subvolume_entry_array().size();
    mesher.MarchingCubes(tsdf_volume);

    std::shared_ptr<TriangleMeshCuda> mesh =
        std::make_shared<TriangleMeshCuda>(mesher.mesh());

    VisualizerWithKeyCallback visualizer;
    if (! visualizer.CreateVisualizerWindow("Visualizer", 640, 480, 0, 0)) {
        PrintWarning("Failed creating OpenGL window.\n");
        return 0;
    }
    visualizer.AddGeometry(mesh);

    visualizer.GetRenderOption().show_coordinate_frame_ = true;
    visualizer.GetRenderOption().mesh_color_option_ =
        RenderOption::MeshColorOption::Normal;
    visualizer.BuildUtilities();
    visualizer.UpdateWindowTitle();

    bool should_close = false;
    while (! should_close) {
        should_close = !visualizer.PollEvents();
    }
    visualizer.DestroyVisualizerWindow();

    WriteTriangleMeshToPLY("test_face.ply", *mesher.mesh().Download());

    return 1;
}
