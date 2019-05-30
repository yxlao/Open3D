//
// Created by wei on 4/28/19.
//

#include <Open3D/Open3D.h>
#include <AdvancedRendering/Visualization/Visualizer/VisualizerPBR.h>
#include <AdvancedRendering/Geometry/ExtendedTriangleMesh.h>
#include <AdvancedRendering/IO/ClassIO/ExtendedTriangleMeshIO.h>
#include <Open3D/Utility/Console.h>
#include <AdvancedRendering/Geometry/ImageExt.h>
#include <AdvancedRendering/Visualization/Utility/DrawGeometryPBR.h>

using namespace open3d;

int main(int argc, char **argv) {
    auto mesh = std::make_shared<geometry::ExtendedTriangleMesh>();
    io::ReadExtendedTriangleMeshFromPLY("/media/wei/Data/data/pbr/model/sphere_plastic.ply", *mesh);
//    for (auto &color : mesh->vertex_colors_) {
//        color = Eigen::Vector3d(1, 1, 0.0);
//    }
//    for (auto &material : mesh->vertex_textures_) {
//        material(1) = 0.0;
//    }

    std::vector<geometry::Image> textures;
    textures.emplace_back(*geometry::FlipImageExt(*io::CreateImageFromFile(
        "/media/wei/Data/data/pbr/image/plastic_alex/40.png")));

    auto target = geometry::FlipImageExt(*io::CreateImageFromFile(
        "/media/wei/Data/data/pbr/image/plastic_alex/40.png"));
    camera::PinholeCameraParameters cam_params;
    io::ReadIJsonConvertibleFromJSON("/media/wei/Data/data/pbr/image/plastic_alex/40.json", cam_params);

    auto ibl = std::make_shared<geometry::IBLLighting>();
    ibl->ReadEnvFromHDR(
        "/media/wei/Data/data/pbr/env/Alexs_Apt_2k.hdr");

//    visualization::DrawGeometriesPBR({mesh}, {textures}, {ibl});

    visualization::VisualizerDR visualizer;
    if (!visualizer.CreateVisualizerWindow("DR", 640, 480, 0, 0)) {
        utility::PrintWarning("Failed creating OpenGL window.\n");
        return 0;
    }
    visualizer.BuildUtilities();
    visualizer.UpdateWindowTitle();

    visualizer.AddGeometryPBR(mesh, textures, ibl);
    float lambda = 0.1;
    for (int i = 0; i < 1; ++i) {
        visualizer.SetTargetImage(*target, cam_params);

        visualizer.UpdateRender();
        visualizer.PollEvents();

        visualizer.CallSGD(lambda, false, false, true);
        if (i % 50 == 49) lambda *= 0.5f;
    }

    visualization::DrawGeometriesPBR({mesh}, {textures}, {ibl});
}