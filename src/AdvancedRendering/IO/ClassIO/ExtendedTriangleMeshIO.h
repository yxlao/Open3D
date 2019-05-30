//
// Created by wei on 4/21/19.
//

#pragma once

#include <AdvancedRendering/Geometry/ExtendedTriangleMesh.h>

namespace open3d {
namespace io {
bool ReadExtendedTriangleMeshFromPLY(const std::string &filename,
                                     geometry::ExtendedTriangleMesh &mesh);
bool WriteExtendedTriangleMeshToPLY(const std::string &filename,
                                    const geometry::ExtendedTriangleMesh &mesh,
                                    bool write_ascii = false,
                                    bool compressed = false);

/** To be separated in another class **/
bool ReadExtendedTriangleMeshFromOBJ(const std::string &filename,
                                     geometry::ExtendedTriangleMesh &mesh);
bool WriteExtendedTriangleMeshToOBJAndImages(
    const std::string &filename, /* filename.obj, filename.mtl, 0, 1, 2.png */
    const geometry::ExtendedTriangleMesh &mesh);
}
}
