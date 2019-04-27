//
// Created by wei on 4/15/19.
//

#include <Open3D/Open3D.h>
#include <InverseRendering/IO/ClassIO/TriangleMeshExtendedIO.h>
#include <InverseRendering/Geometry/TriangleMeshExtended.h>
#include <InverseRendering/Geometry/Lighting.h>
#include <InverseRendering/Visualization/Utility/DrawGeometryPBR.h>

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

    auto ibl = std::make_shared<geometry::IBLLighting>();
    ibl->ReadEnvFromHDR("/media/wei/Data/data/pbr/env/Tokyo_BigSight_3k.hdr");

    utility::SetVerbosityLevel(utility::VerbosityLevel::VerboseDebug);
    visualization::DrawGeometriesPBR({mesh}, {textures}, {ibl});

}
