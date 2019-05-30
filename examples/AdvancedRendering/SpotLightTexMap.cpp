//
// Created by wei on 4/15/19.
//

#include <Open3D/Open3D.h>
#include <AdvancedRendering/Geometry/ExtendedTriangleMesh.h>
#include <AdvancedRendering/Geometry/Lighting.h>
#include <AdvancedRendering/Visualization/Utility/DrawGeometryPBR.h>
#include <AdvancedRendering/Geometry/ImageExt.h>
#include <AdvancedRendering/IO/ClassIO/ExtendedTriangleMeshIO.h>

using namespace open3d;

int main() {
    std::string base_path = "/Users/dongw1/Work/Data/resources/textures/pbr/gold";

    auto mesh = std::make_shared<geometry::ExtendedTriangleMesh>();
    io::ReadExtendedTriangleMeshFromPLY(base_path + "/sphere.ply", *mesh);
    std::vector<std::string> filenames = {base_path + "/albedo.png",
                                          base_path + "/normal.png",
                                          base_path + "/metallic.png",
                                          base_path + "/roughness.png",
                                          base_path + "/ao.png"};
    mesh->LoadImageTextures(filenames);

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
    visualization::DrawGeometriesPBR({mesh}, lighting);
}