//
// Created by wei on 10/10/20.
//

#include <Cuda/Integration/UniformTSDFVolumeCuda.h>
#include <Cuda/Integration/UniformMeshVolumeCuda.h>
#include <Cuda/Geometry/VectorCuda.h>
#include <Core/Core.h>
#include <Eigen/Eigen>
#include <IO/IO.h>

#include "UnitTest.h"

#include <opencv2/opencv.hpp>
#include <vector>

TEST(UniformMeshVolumeCuda, MarchingCubes) {
    using namespace open3d;
    cv::Mat im = cv::imread("../../examples/TestData/RGBD/depth/00000.png",
                            cv::IMREAD_UNCHANGED);
    ImageCuda<open3d::Vector1s> imcuda;
    imcuda.Upload(im);
    auto imcudaf = imcuda.ToFloat(0.001f);

    MonoPinholeCameraCuda default_camera;
    default_camera.SetUp();
    TransformCuda transform = TransformCuda::Identity();

    const float voxel_length = 0.01f;
    transform.SetTranslation(Vector3f(-voxel_length * 256));
    UniformTSDFVolumeCuda<512> volume(voxel_length, voxel_length * 3, transform);

    TransformCuda extrinsics = TransformCuda::Identity();
    volume.Integrate(imcudaf, default_camera, extrinsics);

    UniformMeshVolumeCuda<VertexWithNormal, 512> mesher;
    mesher.Create(100000, 100000);

    Timer timer;
    timer.Start();
    mesher.MarchingCubes(volume);
    timer.Stop();
    PrintInfo("MarchingCubes time: %f milliseconds\n", timer.GetDuration());

    std::shared_ptr<TriangleMesh> mesh = mesher.mesh().Download();
    PrintInfo("triangle.size(): %d, vertices.size(): %d, normals.size(): %d\n",
              mesh->triangles_.size(),
              mesh->vertices_.size(),
              mesh->vertex_normals_.size());
    WriteTriangleMeshToPLY("test.ply", *mesh, true);
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}