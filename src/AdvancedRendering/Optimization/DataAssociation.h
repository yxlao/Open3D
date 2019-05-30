//
// Created by wei on 4/26/19.
//

#pragma once

#include <Open3D/Open3D.h>

namespace open3d {
namespace geometry {

/** Captures per vertex: vector of (frame_idx, pixel_idx) **/
class DataAssociation {
public:
    std::vector<std::vector<std::pair<int, int>>> associated_pixels_;

    DataAssociation(size_t mesh_size) {
        associated_pixels_.resize(mesh_size);
    }

    bool Associate(int frame_idx, const geometry::Image &index_map);

    bool WriteDataAssociationToBin(const std::string &filename);
    bool ReadDataAssociationFromBin(const std::string &filename);
};

}
}


