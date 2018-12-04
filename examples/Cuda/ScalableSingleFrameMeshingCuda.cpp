//
// Created by wei on 10/24/18.
//

#include <Cuda/Integration/ScalableTSDFVolumeCuda.h>
#include <Cuda/Integration/ScalableMeshVolumeCuda.h>
#include <IO/IO.h>
#include <Core/Core.h>

int main(int argc, char **argv) {
    using namespace open3d;
    SetVerbosityLevel(VerbosityLevel::VerboseDebug);

    Image depth, color;
    ReadImage("../../../examples/TestData/RGBD/depth/00000.png", depth);
    ReadImage("../../../examples/TestData/RGBD/color/00000.jpg", color);
    cuda::RGBDImageCuda rgbd(0.1f, 3.5f, 1000.0f);
    rgbd.Upload(depth, color);

    cuda::PinholeCameraIntrinsicCuda intrinsics(
        PinholeCameraIntrinsicParameters::PrimeSenseDefault);

    float voxel_length = 0.01f;
    cuda::TransformCuda extrinsics = cuda::TransformCuda::Identity();
    extrinsics(0, 3) = 10.0f;
    extrinsics(1, 3) = -10.0f;
    extrinsics(2, 3) = 1.0f;
    cuda::ScalableTSDFVolumeCuda<8> tsdf_volume(
        10000, 200000, voxel_length, 3 * voxel_length, extrinsics);

    Timer timer;
    timer.Start();
    int iters = 10;
    for (int i = 0; i < iters; ++i) {
        tsdf_volume.Integrate(rgbd, intrinsics, extrinsics);
    }
    timer.Stop();
    PrintInfo("Integration takes: %f milliseconds\n",
        timer.GetDuration() / iters);

    tsdf_volume.GetAllSubvolumes();
    cuda::ScalableMeshVolumeCuda<8> mesher(
        10000, cuda::VertexWithNormalAndColor, 200000, 400000);

    mesher.active_subvolumes_ = tsdf_volume.active_subvolume_entry_array().size();
    PrintInfo("Active subvolumes: %d\n", mesher.active_subvolumes_);

    timer.Start();
    iters = 100;
    for (int i = 0; i < iters; ++i) {
        mesher.MarchingCubes(tsdf_volume);
    }
    timer.Stop();
    PrintInfo("MarchingCubes takes: %f milliseconds\n",
        timer.GetDuration() / iters);

    WriteTriangleMeshToPLY("test_scalable.ply", *mesher.mesh().Download());
}