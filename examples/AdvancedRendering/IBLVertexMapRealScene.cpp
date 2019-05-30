//
// Created by wei on 4/23/19.
//


#include <Open3D/Open3D.h>
#include <AdvancedRendering/IO/ClassIO/ExtendedTriangleMeshIO.h>
#include <AdvancedRendering/Geometry/ExtendedTriangleMesh.h>
#include <AdvancedRendering/Geometry/Lighting.h>
#include <AdvancedRendering/Visualization/Utility/DrawGeometryPBR.h>
#include <random>

using namespace open3d;

int main() {
    auto mesh = std::make_shared<geometry::TriangleMesh>();
    io::ReadTriangleMeshFromPLY(
        "/media/wei/Data/data/stanford/lounge/scene_cuda/integrated.ply", *mesh);

    auto mesh_extended = std::make_shared<geometry::ExtendedTriangleMesh>();
    mesh_extended->vertices_ = mesh->vertices_;
    mesh_extended->vertex_colors_ = mesh->vertex_colors_;
    mesh_extended->vertex_normals_ = mesh->vertex_normals_;
    mesh_extended->triangles_ = mesh->triangles_;

    std::random_device rd;
    std::uniform_real_distribution<double> dist_roughness(0.8, 1.0);
    std::uniform_real_distribution<double> dist_metallic(0.0, 0.2);
    std::uniform_real_distribution<double> dist_ao(0.8, 1.0);
    mesh_extended->vertex_textures_.resize(mesh->vertices_.size());

    for (auto &mat : mesh_extended->vertex_textures_) {
        mat = Eigen::Vector3d(0, 1, 1);
//        mat = Eigen::Vector3d(dist_roughness(rd), dist_metallic(rd), dist_ao(rd));
    }

    std::vector<geometry::Image> textures; /** dummy **/

    auto ibl = std::make_shared<geometry::IBLLighting>();
    ibl->ReadEnvFromHDR("/media/wei/Data/data/pbr/env/Alexs_Apt_2k.hdr");

    utility::SetVerbosityLevel(utility::VerbosityLevel::VerboseDebug);
    visualization::DrawGeometriesPBR({mesh_extended}, {textures}, {ibl});
}
