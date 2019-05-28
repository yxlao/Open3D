//
// Created by wei on 4/15/19.
//

#include <Open3D/Open3D.h>
#include <InverseRendering/IO/ClassIO/TriangleMeshExtendedIO.h>
#include <InverseRendering/Geometry/TriangleMeshExtended.h>
#include <InverseRendering/Geometry/Lighting.h>
#include <InverseRendering/Visualization/Utility/DrawGeometryPBR.h>
#include <InverseRendering/Geometry/ImageExt.h>

using namespace open3d;

int main() {
    auto mesh = std::make_shared<geometry::TriangleMeshExtended>();
    io::ReadTriangleMeshExtendedFromPLY("/Users/dongw1/Work/Data/resources/textures/pbr/gold/sphere.ply", *mesh);

    std::string base_path = "/Users/dongw1/Work/Data/resources/textures/pbr/gold";
    std::vector<geometry::Image> textures;
    textures.push_back(*io::CreateImageFromFile(base_path + "/albedo.png"));
    textures.push_back(*io::CreateImageFromFile(base_path + "/normal.png"));
    textures.push_back(*io::CreateImageFromFile(base_path + "/metallic.png"));
    textures.push_back(*io::CreateImageFromFile(base_path + "/roughness.png"));
    textures.push_back(*io::CreateImageFromFile(base_path + "/ao.png"));

    auto ibl = std::make_shared<geometry::IBLLighting>();
    ibl->ReadEnvFromHDR("/Users/dongw1/Work/Data/resources/textures/hdr/newport_loft.hdr");

    utility::SetVerbosityLevel(utility::VerbosityLevel::VerboseDebug);
    visualization::DrawGeometriesPBR({mesh}, {textures}, {ibl});
}
