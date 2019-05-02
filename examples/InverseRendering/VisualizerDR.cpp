//
// Created by wei on 4/28/19.
//

#include <Open3D/Open3D.h>
#include <InverseRendering/Visualization/Visualizer/VisualizerPBR.h>
#include <InverseRendering/Geometry/TriangleMeshExtended.h>
#include <InverseRendering/IO/ClassIO/TriangleMeshExtendedIO.h>
#include <Open3D/Utility/Console.h>
#include <InverseRendering/Geometry/ImageExt.h>
#include <InverseRendering/Visualization/Utility/DrawGeometryPBR.h>

using namespace open3d;

int main(int argc, char **argv) {
    auto mesh = std::make_shared<geometry::TriangleMeshExtended>();
    io::ReadTriangleMeshExtendedFromPLY("/media/wei/Data/data/pbr/model/sphere_gold.ply", *mesh);
//    for (auto &color : mesh->vertex_colors_) {
//        color = Eigen::Vector3d(1, 1, 0.0);
//    }
    for (auto &material : mesh->vertex_materials_) {
        material(1) = 0.0;
    }

    std::vector<geometry::Image> textures;
    textures.emplace_back(*geometry::FlipImageExt(*io::CreateImageFromFile(
        "/media/wei/Data/data/pbr/image/gold_alex_apt_buf.png")));

    auto target = geometry::FlipImageExt(*io::CreateImageFromFile("/media/wei/Data/data/pbr/image/gold_01.png"));
    camera::PinholeCameraParameters cam_params;
    io::ReadIJsonConvertibleFromJSON("/media/wei/Data/data/pbr/image/gold_01.json", cam_params);

    auto ibl = std::make_shared<geometry::IBLLighting>();
    ibl->ReadEnvFromHDR(
        "/media/wei/Data/data/pbr/env/Alexs_Apt_2k.hdr");

    visualization::DrawGeometriesPBR({mesh}, {textures}, {ibl});

    visualization::VisualizerDR visualizer;
    if (!visualizer.CreateVisualizerWindow("DR", 640, 480, 0, 0)) {
        utility::PrintWarning("Failed creating OpenGL window.\n");
        return 0;
    }
    visualizer.BuildUtilities();
    visualizer.UpdateWindowTitle();

    visualizer.AddGeometryPBR(mesh, textures, ibl);
    float lambda = 1;
    for (int i = 0; i < 200; ++i) {
        visualizer.SetTargetImage(*target, cam_params);

        visualizer.UpdateRender();
        visualizer.PollEvents();
        visualizer.GetViewControl().ConvertFromPinholeCameraParameters(cam_params);

//        visualizer.CaptureBuffer("/media/wei/Data/data/pbr/image/gold_alex_apt_buf.png");
        visualizer.CallSGD(lambda, false, true, false);
        if (i % 50 == 49) lambda *= 0.5f;
    }

    visualization::DrawGeometriesPBR({mesh}, {textures}, {ibl});
}