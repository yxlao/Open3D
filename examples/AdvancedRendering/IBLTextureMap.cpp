//
// Created by wei on 4/15/19.
//

#include <Open3D/Open3D.h>
#include <AdvancedRendering/IO/ClassIO/TexturedTriangleMeshIO.h>
#include <AdvancedRendering/Geometry/TexturedTriangleMesh.h>
#include <AdvancedRendering/Geometry/Lighting.h>
#include <AdvancedRendering/Visualization/Utility/DrawGeometryPBR.h>
#include <AdvancedRendering/Geometry/ImageExt.h>

using namespace open3d;

int main() {
    std::string base_path = "/Users/dongw1/Work/Data/planet";

    auto mesh = std::make_shared<geometry::TexturedTriangleMesh>();
    io::ReadTexturedTriangleMeshFromOBJ(base_path + "/planet.obj", *mesh);

    auto ibl = std::make_shared<geometry::IBLLighting>();
    ibl->ReadEnvFromHDR(
        "/Users/dongw1/Work/Data/resources/textures/hdr/newport_loft.hdr");

    utility::SetVerbosityLevel(utility::VerbosityLevel::VerboseDebug);
    visualization::DrawGeometriesPBR({mesh}, ibl);
}
