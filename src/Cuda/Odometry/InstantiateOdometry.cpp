//
// Created by wei on 11/9/18.
//

#include "RGBDOdometryCudaHost.hpp"

#include <src/Cuda/Geometry/ImageCudaHost.hpp>

namespace open3d {
namespace cuda {
template class RGBDOdometryCuda<3>;

template class ImageCuda<float, 6>;
} // cuda
} // open3d