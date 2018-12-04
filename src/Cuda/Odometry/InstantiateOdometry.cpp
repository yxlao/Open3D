//
// Created by wei on 11/9/18.
//

#include "RGBDOdometryCudaHost.hpp"
#include "ICRGBDOdometryCudaHost.hpp"

#include <Cuda/Geometry/ImageCudaHost.hpp>
#include <Cuda/Geometry/ImagePyramidCudaHost.hpp>

namespace open3d {
namespace cuda {
template class RGBDOdometryCuda<3>;
template class ICRGBDOdometryCuda<3>;

template class ImageCuda<Vector6f>;
template class ImagePyramidCuda<Vector6f, 3>;
} // cuda
} // open3d