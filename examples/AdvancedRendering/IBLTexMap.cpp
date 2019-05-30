//
// Created by wei on 4/15/19.
//

#include <Open3D/Open3D.h>
#include <AdvancedRendering/IO/ClassIO/ExtendedTriangleMeshIO.h>
#include <AdvancedRendering/Geometry/ExtendedTriangleMesh.h>
#include <AdvancedRendering/Geometry/Lighting.h>
#include <AdvancedRendering/Visualization/Utility/DrawGeometryPBR.h>
#include <AdvancedRendering/Geometry/ImageExt.h>

using namespace open3d;

int main() {
    std::string base_path = "/Users/dongw1/Work/Data/resources/textures/pbr/rusted_iron";

    auto mesh = std::make_shared<geometry::ExtendedTriangleMesh>();
    io::ReadExtendedTriangleMeshFromPLY(base_path + "/sphere.ply", *mesh);
    utility::PrintInfo("%d %d\n", mesh->HasVertexNormals(), mesh->HasUVs());

    std::vector<std::string> filenames = {base_path + "/albedo.png",
                                          base_path + "/normal.png",
                                          base_path + "/metallic.png",
                                          base_path + "/roughness.png",
                                          base_path + "/ao.png"};
    mesh->LoadImageTextures(filenames);

    auto ibl = std::make_shared<geometry::IBLLighting>();
    ibl->ReadEnvFromHDR(
        "/Users/dongw1/Work/Data/resources/textures/hdr/newport_loft.hdr");

    utility::SetVerbosityLevel(utility::VerbosityLevel::VerboseDebug);
    visualization::DrawGeometriesPBR({mesh}, ibl);
}
