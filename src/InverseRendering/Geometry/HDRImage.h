//
// Created by wei on 4/23/19.
//

#pragma once

#include <Open3D/Open3D.h>
#include <stb_image/stb_image.h>
#include <stb_image/stb_image_write.h>

namespace open3d {
namespace geometry {
class HDRImage {
public:
    bool ReadFromHDR(const std::string &filename, bool flip = true);
    bool WriteToHDR(const std::string &filename, bool flip = true);

    std::shared_ptr<Image> image_;
};
}
}


