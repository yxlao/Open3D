//
// Created by wei on 4/15/19.
//

#include <Open3D/Open3D.h>
#include <InverseRendering/IO/ClassIO/ExtendedTriangleMeshIO.h>
#include <InverseRendering/Geometry/ExtendedTriangleMesh.h>
#include <InverseRendering/Geometry/Lighting.h>
#include <InverseRendering/Visualization/Utility/DrawGeometryPBR.h>
#include <InverseRendering/Geometry/ImageExt.h>

using namespace open3d;

int main() {
    auto mesh = std::make_shared<geometry::ExtendedTriangleMesh>();
    io::ReadExtendedTriangleMeshFromPLY("/Users/dongw1/Work/Data/resources/textures/pbr/gold/sphere.ply", *mesh);


    std::string base_path = "/Users/dongw1/Work/Data/resources/textures/pbr/gold";

    std::vector<std::string> filenames = {base_path + "/albedo.png",
                                          base_path + "/normal.png",
                                          base_path + "/metallic.png",
                                          base_path + "/roughness.png",
                                          base_path + "/ao.png"};
    mesh->LoadImageTextures(filenames);

    auto ibl = std::make_shared<geometry::IBLLighting>();
    ibl->ReadEnvFromHDR("/Users/dongw1/Work/Data/resources/textures/hdr/newport_loft.hdr");

    utility::SetVerbosityLevel(utility::VerbosityLevel::VerboseDebug);
    visualization::DrawGeometriesPBR({mesh}, {ibl});
}
