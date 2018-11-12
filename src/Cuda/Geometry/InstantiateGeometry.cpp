//
// Created by wei on 11/9/18.
//

#include "ImageCudaHost.hpp"
#include "RGBDImageCudaHost.hpp"

#include "ImagePyramidCudaHost.hpp"
#include "RGBDImagePyramidCudaHost.hpp"

#include "TriangleMeshCudaHost.hpp"

namespace open3d {

template class ImageCuda<Vector1s>;
template class ImageCuda<Vector4b>;
template class ImageCuda<Vector3b>;
template class ImageCuda<Vector1b>;
template class ImageCuda<Vector4f>;
template class ImageCuda<Vector3f>;
template class ImageCuda<Vector1f>;

template class ImagePyramidCuda<Vector1s, 3>;
template class ImagePyramidCuda<Vector1s, 4>;
template class ImagePyramidCuda<Vector4b, 3>;
template class ImagePyramidCuda<Vector4b, 4>;
template class ImagePyramidCuda<Vector3b, 3>;
template class ImagePyramidCuda<Vector3b, 4>;
template class ImagePyramidCuda<Vector1b, 3>;
template class ImagePyramidCuda<Vector1b, 4>;
template class ImagePyramidCuda<Vector4f, 3>;
template class ImagePyramidCuda<Vector4f, 4>;
template class ImagePyramidCuda<Vector3f, 3>;
template class ImagePyramidCuda<Vector3f, 4>;
template class ImagePyramidCuda<Vector1f, 3>;
template class ImagePyramidCuda<Vector1f, 4>;

template class RGBDImagePyramidCuda<3>;
template class RGBDImagePyramidCuda<4>;
}