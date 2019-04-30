//
// Created by wei on 4/28/19.
//

#include <Open3D/Open3D.h>
#include <InverseRendering/Visualization/Visualizer/VisualizerPBR.h>
#include <InverseRendering/Geometry/TriangleMeshExtended.h>
#include <InverseRendering/IO/ClassIO/TriangleMeshExtendedIO.h>
#include <Open3D/Utility/Console.h>
#include <InverseRendering/Geometry/ImageExt.h>

using namespace open3d;

int main(int argc, char **argv) {
    auto mesh = std::make_shared<geometry::TriangleMeshExtended>();
    io::ReadTriangleMeshExtendedFromPLY("/media/wei/Data/data/pbr/model/sphere_plastic.ply", *mesh);
    for (auto &color : mesh->vertex_colors_) {
        color = Eigen::Vector3d(1, 1, 0.0);
    }
    std::vector<geometry::Image> textures; /** dummy **/
    textures.emplace_back(*io::ReadImageFromHDR("/media/wei/Data/data/pbr/image/plastic_alex_apt.hdr"));

    auto ibl = std::make_shared<geometry::IBLLighting>();
    ibl->ReadEnvFromHDR("/media/wei/Data/data/pbr/env/Alexs_Apt_2k.hdr");

    visualization::VisualizerPBR visualizer;
    if (!visualizer.CreateVisualizerWindow("PBR", 640, 480, 0, 0)) {
        utility::PrintWarning("Failed creating OpenGL window.\n");
        return 0;
    }
    visualizer.BuildUtilities();
    visualizer.UpdateWindowTitle();

    visualizer.AddGeometryPBR(mesh, textures, ibl);

    float lambda = 0.01;
    for (int i = 0; i < 1000; ++i) {
        /* render to buffer */
        visualizer.UpdateGeometry();
        visualizer.PollEvents();

        auto diffs = visualizer.geometry_renderer_fbo_outputs_[0];
        auto index_map = diffs[8];
        auto normal_map = diffs[7];

        auto d_albedo = diffs[2];
        auto d_color = diffs[6];
        for (int v = 0; v < index_map->height_; ++v) {
            for (int u = 0; u < index_map->width_; ++u) {
                int *idx = geometry::PointerAt<int>(*index_map, u, v);
                if (*idx > 0) {
                    float *d_albedo_r = geometry::PointerAt<float>(*d_albedo, u, v, 0);
                    float *d_albedo_g = geometry::PointerAt<float>(*d_albedo, u, v, 1);
                    float *d_albedo_b = geometry::PointerAt<float>(*d_albedo, u, v, 2);
                    Eigen::Vector3d d_albedo = Eigen::Vector3d(
                        *d_albedo_r, *d_albedo_g, *d_albedo_b);

                    float *d_color_r = geometry::PointerAt<float>(*d_color, u, v, 0);
                    float *d_color_g = geometry::PointerAt<float>(*d_color, u, v, 1);
                    float *d_color_b = geometry::PointerAt<float>(*d_color, u, v, 2);
                    Eigen::Vector3d d_color = Eigen::Vector3d(
                        *d_color_r, *d_color_g, *d_color_b);

//                    std::cout << d_albedo.transpose() << " -- " << d_color.transpose() << "\n";
                    auto &color = mesh->vertex_colors_[*idx];
                    color(0) -= lambda * (*d_albedo_r * *d_color_r);
                    color(1) -= lambda * (*d_albedo_g * *d_color_g);
                    color(2) -= lambda * (*d_albedo_b * *d_color_b);
                }
            }
        }
    }
}