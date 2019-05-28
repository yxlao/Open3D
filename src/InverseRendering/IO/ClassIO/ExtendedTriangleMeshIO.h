//
// Created by wei on 4/21/19.
//

#pragma once

#include <InverseRendering/Geometry/ExtendedTriangleMesh.h>

namespace open3d {
namespace io {
bool ReadExtendedTriangleMeshFromPLY(const std::string &filename,
                                     geometry::ExtendedTriangleMesh &mesh);
bool WriteExtendedTriangleMeshToPLY(const std::string &filename,
                                    const geometry::ExtendedTriangleMesh &mesh,
                                    bool write_ascii = false,
                                    bool compressed = false);
}
}
