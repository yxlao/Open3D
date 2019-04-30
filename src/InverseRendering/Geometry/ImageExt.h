//
// Created by wei on 4/27/19.
//

#pragma once

#include <Open3D/Open3D.h>

namespace open3d {
namespace geometry {

std::shared_ptr<Image> ConvertImageFromFloatImage(const Image &image);
}

namespace io {
bool WriteImageToHDR(const std::string &filename, const geometry::Image &image);
std::shared_ptr<geometry::Image> ReadImageFromHDR(const std::string &filename);

}
}