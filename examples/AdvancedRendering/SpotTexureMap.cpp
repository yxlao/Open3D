//
// Created by wei on 4/15/19.
//

#include <Open3D/Open3D.h>
#include <AdvancedRendering/Geometry/TexturedTriangleMesh.h>
#include <AdvancedRendering/Geometry/Lighting.h>
#include <AdvancedRendering/Visualization/Utility/DrawGeometryPBR.h>
#include <AdvancedRendering/Geometry/ImageExt.h>
#include <AdvancedRendering/IO/ClassIO/TexturedTriangleMeshIO.h>
#include "data_path.h"

using namespace open3d;

int main() {
    std::string base_path = kBasePath;

    auto mesh = std::make_shared<geometry::TexturedTriangleMesh>();
    io::ReadTexturedTriangleMeshFromOBJ(base_path + "/planet/planet.obj", *mesh);

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