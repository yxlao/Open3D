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
    auto mesh = std::make_shared<geometry::ExtendedTriangleMesh>();
    io::ReadExtendedTriangleMeshFromPLY(
//        "mesh-iter-99.ply", *mesh);
        "/Users/dongw1/Work/Data/resources/textures/pbr/rusted_iron/sphere.ply", *mesh);
//        "/media/wei/Data/data/pbr/model/sphere_gold.ply", *mesh);

    auto ibl = std::make_shared<geometry::IBLLighting>();
    ibl->ReadEnvFromHDR("/Users/dongw1/Work/Data/resources/textures/hdr/newport_loft.hdr");
//    ibl->ReadEnvFromHDR("/media/wei/Data/data/pbr/env/Alexs_Apt_2k.hdr");
    
    utility::SetVerbosityLevel(utility::VerbosityLevel::VerboseDebug);
    visualization::DrawGeometriesPBR({mesh}, ibl);
}
