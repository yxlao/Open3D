//
// Created by wei on 4/21/19.
//

#pragma once

#include <InverseRendering/Geometry/TriangleMeshExtended.h>

namespace open3d {
namespace io {
bool ReadTriangleMeshExtendedFromPLY(const std::string &filename,
                                     geometry::TriangleMeshExtended &mesh);
bool WriteTriangleMeshExtendedToPLY(const std::string &filename,
                                    const geometry::TriangleMeshExtended &mesh,
                                    bool write_ascii = false,
                                    bool compressed = false);
}
}
