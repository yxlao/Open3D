//
// Created by wei on 4/23/19.
//


#include <Open3D/Open3D.h>
#include <Material/IO/ClassIO/TriangleMeshExtendedIO.h>
#include <Material/Physics/TriangleMeshExtended.h>
#include <Material/Physics/Lighting.h>
#include <Material/Visualization/Utility/DrawGeometryPBR.h>
#include <random>

using namespace open3d;
int main() {
    auto mesh = std::make_shared<geometry::TriangleMesh>();
    io::ReadTriangleMeshFromPLY(
        "/media/wei/Data/data/stanford/lounge/scene_cuda/integrated.ply", *mesh);

    auto mesh_extended = std::make_shared<geometry::TriangleMeshExtended>();
    mesh_extended->vertices_ = mesh->vertices_;
    mesh_extended->vertex_colors_ = mesh->vertex_colors_;
    mesh_extended->vertex_normals_ = mesh->vertex_normals_;
    mesh_extended->triangles_ = mesh->triangles_;

    std::random_device rd;
    std::uniform_real_distribution<double> dist_roughness(0.8, 1.0);
    std::uniform_real_distribution<double> dist_metallic(0.0, 0.2);
    std::uniform_real_distribution<double> dist_ao(0.8, 1.0);
    mesh_extended->vertex_materials_.resize(mesh->vertices_.size());

    for (auto &mat : mesh_extended->vertex_materials_) {
        mat = Eigen::Vector3d(0.7, 0, 1);
//        mat = Eigen::Vector3d(dist_roughness(rd), dist_metallic(rd), dist_ao(rd));
    }

    std::vector<geometry::Image> textures; /** dummy **/

    auto ibl = std::make_shared<physics::IBLLighting>();
    ibl->filename_ = "/media/wei/Data/data/pbr/env/White.hdr";

    utility::SetVerbosityLevel(utility::VerbosityLevel::VerboseDebug);
    visualization::DrawGeometriesPBR({mesh_extended}, {textures}, {ibl});
}
