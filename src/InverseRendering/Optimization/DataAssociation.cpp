//
// Created by wei on 4/26/19.
//

#include "DataAssociation.h"
#include <fstream>

namespace open3d {
namespace geometry {

bool DataAssociation::Associate(int frame_idx,
                                const geometry::Image &index_map) {
    for (int u = 0; u < index_map.width_; ++u) {
        for (int v = 0; v < index_map.height_; ++v) {
            int *idx = geometry::PointerAt<int>(index_map, u, v);
            if (*idx != 0) {
                associated_pixels_[*idx].emplace_back(
                        std::make_pair(frame_idx, v * index_map.width_ + u));
            }
        }
    }
    return true;
}

bool DataAssociation::ReadDataAssociationFromBin(const std::string &filename) {
    return false;
}

bool DataAssociation::WriteDataAssociationToBin(const std::string &filename) {
    FILE *fid = fopen(filename.c_str(), "wb");
    if (fid == NULL) {
        utility::PrintError("Unable to create file: %f\n", filename.c_str());
        return false;
    }

    int num_points = associated_pixels_.size();
    if (fwrite(&num_points, sizeof(int), 1, fid) < 1) {
        utility::PrintError("Write BIN failed: unable to write num points\n");
        return false;
    }

    for (int i = 0; i < num_points; ++i) {
        auto &vec = associated_pixels_[i];

        int num_associations = vec.size();
        if (fwrite(&num_associations, sizeof(int), 1, fid) < 1) {
            utility::PrintError(
                    "Write BIN failed: unable to write num assocs\n");
            return false;
        }

        if (fwrite(vec.data(), sizeof(std::pair<int, int>), num_associations,
                   fid) < num_associations) {
            utility::PrintError("Write BIN failed: unable to write assocs\n");
            return false;
        }
    }

    return true;
}

}  // namespace geometry
}  // namespace open3d
