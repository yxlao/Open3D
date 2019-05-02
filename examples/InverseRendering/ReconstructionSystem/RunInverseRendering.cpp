//
// Created by wei on 2/4/19.
//

#include <InverseRendering/Geometry/TriangleMeshExtended.h>
#include <InverseRendering/Geometry/ImageExt.h>
#include <InverseRendering/Geometry/Lighting.h>
#include <InverseRendering/Visualization/Visualizer/VisualizerPBR.h>
#include <Open3D/Registration/PoseGraph.h>
#include <InverseRendering/Visualization/Utility/DrawGeometryPBR.h>
#include "DatasetConfig.h"

#include <Eigen/Eigen>
#include <InverseRendering/IO/ClassIO/TriangleMeshExtendedIO.h>

using namespace open3d;
using namespace open3d::io;
using namespace open3d::utility;

int main(int argc, char **argv) {
    DatasetConfig config;

    std::string config_path = argc > 1 ? argv[1] :
        kDefaultDatasetConfigDir + "/stanford/lounge.json";

    bool is_success = ReadIJsonConvertible(config_path, config);
    if (!is_success) return 1;

    auto mesh = std::make_shared<geometry::TriangleMesh>();
    io::ReadTriangleMeshFromPLY(config.GetPlyFileForFragment(1), *mesh);

    auto mesh_extended = std::make_shared<geometry::TriangleMeshExtended>();
    mesh_extended->vertices_ = mesh->vertices_;
    mesh_extended->vertex_colors_ = mesh->vertex_colors_;
    mesh_extended->vertex_normals_ = mesh->vertex_normals_;
    mesh_extended->triangles_ = mesh->triangles_;
    mesh_extended->vertex_materials_.resize(mesh->vertices_.size());
    for (auto &mat : mesh_extended->vertex_materials_) {
        mat = Eigen::Vector3d(1, 0, 1);
    }

    std::vector<geometry::Image> textures;
    textures.emplace_back(*geometry::FlipImageExt(
        *io::CreateImageFromFile(config.color_files_[0])));

    auto ibl = std::make_shared<geometry::IBLLighting>();
    ibl->ReadEnvFromHDR("/media/wei/Data/data/pbr/env/White.hdr");
    visualization::DrawGeometriesPBR({mesh_extended}, {textures}, {ibl});

    auto mesh_extended_after = std::make_shared<geometry::TriangleMeshExtended>();
    io::ReadTriangleMeshExtendedFromPLY("fragment_extended.ply", *mesh_extended_after);
    visualization::DrawGeometriesPBR({mesh_extended_after}, {textures}, {ibl});
    return 0;

    visualization::VisualizerDR visualizer;
    if (!visualizer.CreateVisualizerWindow("DR", 640, 480, 0, 0)) {
        utility::PrintWarning("Failed creating OpenGL window.\n");
        return 0;
    }
    visualizer.BuildUtilities();
    visualizer.UpdateWindowTitle();

    visualizer.AddGeometryPBR(mesh_extended, textures, ibl);

    camera::PinholeCameraParameters cam_params;
    cam_params.intrinsic_ = camera::PinholeCameraIntrinsic(
        camera::PinholeCameraIntrinsicParameters::PrimeSenseDefault);

    registration::PoseGraph local_pose_graph;
    ReadPoseGraph(config.GetPoseGraphFileForFragment(1, true), local_pose_graph);

    const int iter = 10;
    float lambda = 0.005;

    for (int i = 0; i < iter; ++i) {
        float loss = 0;
        for (int j = 100; j < 200; ++j) {
            auto target = geometry::FlipImageExt(*io::CreateImageFromFile(config.color_files_[j]));
            cam_params.extrinsic_ = local_pose_graph.nodes_[j - 100].pose_.inverse();
            visualizer.SetTargetImage(*target, cam_params);

            visualizer.UpdateRender();
            visualizer.PollEvents();

            loss += visualizer.CallSGD(lambda, false, true, false);
        }
        utility::PrintInfo("Iter %d: lambda = %f -> loss = %f\n",
                           i, lambda, loss);

//        if (i % 10 == 9) {
//            lambda *= 0.1f;
//            io::WriteTriangleMeshExtendedToPLY(
//                "mesh-iter-" + std::to_string(i) + ".ply", *mesh);
//        }
    }

    visualization::DrawGeometriesPBR({mesh_extended}, {textures}, {ibl});
    io::WriteTriangleMeshExtendedToPLY("fragment_extended.ply", *mesh_extended);
}