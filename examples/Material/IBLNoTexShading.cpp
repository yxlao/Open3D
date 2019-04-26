//
// Created by wei on 4/15/19.
//

#include <Open3D/Open3D.h>
#include <Material/IO/ClassIO/TriangleMeshExtendedIO.h>
#include <Material/Physics/TriangleMeshExtended.h>
#include <Material/Physics/Lighting.h>
#include <Material/Visualization/Utility/DrawGeometryPBR.h>

using namespace open3d;
int main() {
    auto mesh = std::make_shared<geometry::TriangleMeshExtended>();
    io::ReadTriangleMeshExtendedFromPLY(
        "/media/wei/Data/data/pbr/model/sphere_wall.ply", *mesh);

    std::vector<geometry::Image> textures; /** dummy **/

    auto ibl = std::make_shared<physics::IBLLighting>();
    ibl->ReadEnvFromHDR("/media/wei/Data/data/pbr/env/Mans_Outside_2k.hdr");

    utility::SetVerbosityLevel(utility::VerbosityLevel::VerboseDebug);
    visualization::DrawGeometriesPBR({mesh}, {textures}, {ibl});
}
