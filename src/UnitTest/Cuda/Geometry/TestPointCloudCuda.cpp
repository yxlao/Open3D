//
// Created by wei on 11/13/18.
//

#include <gtest/gtest.h>
#include <Open3D/Open3D.h>
#include <Cuda/Geometry/RGBDImageCuda.h>
#include <Cuda/Geometry/PointCloudCuda.h>

using namespace open3d;
using namespace open3d::utility;
using namespace open3d::io;
using namespace open3d::cuda;

TEST(PointCloudCuda, UploadAndDownload) {
    using namespace open3d;
    const std::string kDepthPath = "../../../examples/TestData/RGBD/depth/00000.png";
    const std::string kColorPath = "../../../examples/TestData/RGBD/color/00000.jpg";

    geometry::Image depth, color;
    ReadImage(kDepthPath, depth);
    ReadImage(kColorPath, color);

    RGBDImageCuda rgbd_image(640, 480, 3.0f, 1000.0f);
    rgbd_image.Upload(depth, color);

    PinholeCameraIntrinsicCuda intrinsic = PinholeCameraIntrinsicCuda
        (camera::PinholeCameraIntrinsicParameters::PrimeSenseDefault);

    PointCloudCuda pcl(VertexWithColor, 300000);
    pcl.Build(rgbd_image, intrinsic);

    WritePointCloudToPLY("test_pcl.ply", *pcl.Download(), true);
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}