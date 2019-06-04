//
// Created by wei on 4/15/19.
//

#include <Open3D/Open3D.h>

#include <AdvancedRendering/Geometry/TexturedTriangleMesh.h>
#include <AdvancedRendering/IO/ClassIO/TexturedTriangleMeshIO.h>

#include <AdvancedRendering/Visualization/Utility/DrawGeometryUV.h>

using namespace open3d;

int main() {
    std::string base_path = "/Users/dongw1/Work/Data/resources/objects/cyborg";

    auto mesh = std::make_shared<geometry::TexturedTriangleMesh>();
    io::ReadTexturedTriangleMeshFromOBJ(base_path + "/cyborg.obj", *mesh);

    auto target = io::CreateImageFromFile(base_path + "/capture.png");
    visualization::DrawGeometriesUV({mesh}, true);
}
