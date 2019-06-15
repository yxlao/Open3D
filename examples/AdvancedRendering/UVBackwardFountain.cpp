
//
// Created by wei on 4/15/19.
//

#include <chrono>
#include <fstream>
#include <iomanip>
#include <thread>

#include <Open3D/Open3D.h>

#include <AdvancedRendering/Geometry/TexturedTriangleMesh.h>
#include <AdvancedRendering/IO/ClassIO/TexturedTriangleMeshIO.h>
#include <AdvancedRendering/Visualization/Visualizer/VisualizerUV.h>

#include "data_path.h"

using namespace open3d;
using namespace open3d::visualization;

std::vector<std::string> LoadKeyImageNames(const std::string &image_path,
                                           const std::string &key_image_txt) {
    std::vector<std::string> image_filenames;
    utility::filesystem::ListFilesInDirectoryWithExtension(image_path, "jpg",
                                                           image_filenames);
    std::sort(image_filenames.begin(), image_filenames.end());

    std::vector<std::string> key_filenames;
    std::ifstream in(key_image_txt);
    std::vector<int> key_indices;
    if (!in.is_open()) {
        utility::PrintError("Unable to open key image file, abort\n");
        return key_filenames;
    }

    int index;
    while (in >> index) {
        std::string filename = image_filenames[index - 1];
        key_filenames.emplace_back(filename);
    }
    return key_filenames;
}

int main() {
    std::string base_path = kStanfordBasePath + "/fountain";

    /** Load keyframes **/
    auto key_filenames =
            LoadKeyImageNames(base_path + "/image", base_path + "/key.txt");

    /** Load 3D object **/
    int target_tex_width = 2048;
    int target_tex_height = 2048;
    auto mesh_obj = std::make_shared<geometry::TexturedTriangleMesh>();
    io::ReadTexturedTriangleMeshFromOBJ(base_path + "/fountain-10k.obj",
                                        *mesh_obj, target_tex_width,
                                        target_tex_height);

    /** Correct the built-in transform from blender **/
    Eigen::Matrix4d transform;
    transform << 1, 0, 0, 0, 0, 0, -1, 0, 0, 1, 0, 0, 0, 0, 0, 1;
    mesh_obj->Transform(transform);

    /** Load intrinsics **/
    camera::PinholeCameraIntrinsic intrinsic(1280, 1024, 1050.0, 1050.0, 639.5,
                                             511.5);

    /** Load trajectory **/
    camera::PinholeCameraTrajectory traj_key;
    io::ReadPinholeCameraTrajectoryFromLOG(base_path + "/fountain_key.log",
                                           traj_key);
    for (auto &pose : traj_key.parameters_) {
        pose.intrinsic_ = intrinsic;
    }

    /** Build visualizer **/
    VisualizerUV visualizer;
    if (!visualizer.CreateVisualizerWindow("test", 1280, 1024, 0, 0)) {
        utility::PrintWarning(
                "[DrawGeometriesUV] Failed creating OpenGL window.\n");
        return -1;
    }
    visualizer.AddGeometry(mesh_obj);

    for (int i = 0; i < key_filenames.size(); ++i) {
        auto target = io::CreateImageFromFile(key_filenames[i]);
        visualizer.EnableBackwardMode(target);

        camera::PinholeCameraParameters params;
        params.intrinsic_ = intrinsic;
        params.extrinsic_ = traj_key.parameters_[i].extrinsic_;
        visualizer.GetViewControl().ConvertFromPinholeCameraParameters(params);

        visualizer.UpdateRender();
        visualizer.PollEvents();
        visualizer.UpdateSumTextures();

        auto pair = visualizer.GetSumTextures();
//        io::WriteImage("origin-color-" + std::to_string(i) + ".png", *target);
        io::WriteImage("delta-color-" + std::to_string(i) + ".png",
                       *geometry::ConvertImageFromFloatImage(*pair.first));
        io::WriteImage("delta-weight-" + std::to_string(i) + ".png",
                       *geometry::ConvertImageFromFloatImage(*pair.second));

        std::this_thread::sleep_for(std::chrono::milliseconds(1000));
    }

    visualizer.DestroyVisualizerWindow();
    return 0;
}
