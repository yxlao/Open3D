//
// Created by wei on 4/15/19.
//

#include <Open3D/Open3D.h>
#include <InverseRendering/Geometry/TriangleMeshExtended.h>
#include <InverseRendering/Geometry/Lighting.h>
#include <InverseRendering/Visualization/Utility/DrawGeometryPBR.h>
#include <InverseRendering/Geometry/ImageExt.h>
#include <InverseRendering/IO/ClassIO/TriangleMeshExtendedIO.h>

using namespace open3d;

int main() {
    auto mesh = std::make_shared<geometry::TriangleMeshExtended>();
    io::ReadTriangleMeshExtendedFromPLY("/media/wei/Data/data/pbr/model/sphere_uv.ply", *mesh);

    std::string base_path = "/media/wei/Data/data/pbr/materials/plastic";
    std::vector<geometry::Image> textures;
    textures.push_back(*io::CreateImageFromFile(base_path + "/albedo.png"));
    textures.push_back(*io::CreateImageFromFile(base_path + "/normal.png"));
    textures.push_back(*io::CreateImageFromFile(base_path + "/metallic.png"));
    textures.push_back(*io::CreateImageFromFile(base_path + "/roughness.png"));
    textures.push_back(*io::CreateImageFromFile(base_path + "/ao.png"));

    auto lighting = std::make_shared<geometry::SpotLighting>();
    lighting->light_positions_ = {
        Eigen::Vector3f(0, 0, 10),
        Eigen::Vector3f(10, 0, 0),
        Eigen::Vector3f(0, 10, 0)
    };
    lighting->light_colors_ = {
        Eigen::Vector3f(150, 150, 150),
        Eigen::Vector3f(150, 150, 150),
        Eigen::Vector3f(150, 150, 150)
    };

    utility::SetVerbosityLevel(utility::VerbosityLevel::VerboseDebug);
    visualization::DrawGeometriesPBR({mesh}, {textures}, {lighting});
}