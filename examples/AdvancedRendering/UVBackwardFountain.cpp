//
// Created by wei on 4/15/19.
//

#include <Open3D/Open3D.h>

#include <AdvancedRendering/Geometry/TexturedTriangleMesh.h>
#include <AdvancedRendering/IO/ClassIO/TexturedTriangleMeshIO.h>

#include <AdvancedRendering/Visualization/Visualizer/VisualizerUV.h>

using namespace open3d;
using namespace open3d::visualization;

int main() {
    std::string base_path = "/Users/dongw1/Work/Data/fountain";

    auto mesh_obj = std::make_shared<geometry::TexturedTriangleMesh>();
    io::ReadTexturedTriangleMeshFromOBJ(base_path + "/fountain-10k.obj", *mesh_obj);
    Eigen::Matrix4d transform;
    transform << 1, 0, 0, 0,
                0, 0, -1, 0,
                0, 1, 0, 0,
                0, 0, 0, 1;
    mesh_obj->Transform(transform);

    camera::PinholeCameraIntrinsic intrinsic(
        1280, 1024, 1050.0, 1050.0, 639.5, 511.5);
    camera::PinholeCameraTrajectory traj_key;
    io::ReadPinholeCameraTrajectoryFromLOG(
        base_path + "/fountain_all/fountain_key.log", traj_key);
    for (auto &pose : traj_key.parameters_) {
        pose.intrinsic_ = intrinsic;
    }

    VisualizerUV visualizer;

    if (!visualizer.CreateVisualizerWindow(
        "test", 1280, 1024, 0, 0)) {
        utility::PrintWarning(
            "[DrawGeometriesUV] Failed creating OpenGL window.\n");
        return false;
    }
    visualizer.AddGeometry(mesh_obj);

    auto target = io::CreateImageFromFile(base_path + "/9.jpg");
    visualizer.Setup(false, target);

    camera::PinholeCameraParameters params;
    params.intrinsic_ = intrinsic;
    params.extrinsic_ = traj_key.parameters_[0].extrinsic_;
//         traj_key.parameters_.size() - 1].extrinsic_;
    visualizer.GetViewControl().ConvertFromPinholeCameraParameters(params);

//
//    visualizer
//        .GetViewControl()
//        .ConvertFromPinholeCameraParameters(traj_key.parameters_[0]);

    visualizer.Run();
    visualizer.DestroyVisualizerWindow();
}
