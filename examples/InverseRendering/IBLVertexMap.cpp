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
    io::ReadExtendedTriangleMeshFromPLY(
//        "mesh-iter-99.ply", *mesh);
        "/Users/dongw1/Work/Data/resources/textures/pbr/gold/sphere.ply", *mesh);
//        "/media/wei/Data/data/pbr/model/sphere_gold.ply", *mesh);

    auto ibl = std::make_shared<geometry::IBLLighting>();
    ibl->ReadEnvFromHDR("/Users/dongw1/Work/Data/resources/textures/hdr/newport_loft.hdr");
//    ibl->ReadEnvFromHDR("/media/wei/Data/data/pbr/env/Alexs_Apt_2k.hdr");
    
    utility::SetVerbosityLevel(utility::VerbosityLevel::VerboseDebug);
    visualization::DrawGeometriesPBR({mesh}, ibl);
}
