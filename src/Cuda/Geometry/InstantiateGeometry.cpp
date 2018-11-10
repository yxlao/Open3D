//
// Created by wei on 11/9/18.
//

#include "ImageCudaHost.hpp"

namespace open3d {

template class ImageCuda<Vector1s>;
template class ImageCuda<Vector4b>;
template class ImageCuda<Vector3b>;
template class ImageCuda<Vector1b>;
template class ImageCuda<Vector4f>;
template class ImageCuda<Vector3f>;
template class ImageCuda<Vector1f>;

}