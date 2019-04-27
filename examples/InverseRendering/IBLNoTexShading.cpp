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
    io::ReadTriangleMeshExtendedFromPLY(
        "/media/wei/Data/data/pbr/model/sphere_grass.ply", *mesh);

    std::vector<geometry::Image> textures; /** dummy **/

    auto ibl = std::make_shared<geometry::IBLLighting>();
    ibl->ReadEnvFromHDR("/media/wei/Data/data/pbr/env/Mans_Outside_2k.hdr");

    utility::SetVerbosityLevel(utility::VerbosityLevel::VerboseDebug);
    visualization::DrawGeometriesPBR({mesh}, {textures}, {ibl});
}
