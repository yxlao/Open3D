//
// Created by wei on 11/9/18.
//

#include "ImageCudaHost.hpp"
#include "RGBDImageCudaHost.hpp"

#include "ImagePyramidCudaHost.hpp"
#include "RGBDImagePyramidCudaHost.hpp"

#include "PointCloudCudaHost.hpp"
#include "TriangleMeshCudaHost.hpp"

#include "NNCudaHost.hpp"

namespace open3d {

namespace cuda {
template
class ImageCuda<ushort, 1>;
template
class ImageCuda<uchar, 4>;
template
class ImageCuda<uchar, 3>;
template
class ImageCuda<uchar, 1>;
template
class ImageCuda<float, 4>;
template
class ImageCuda<float, 3>;
template
class ImageCuda<float, 1>;

template
class ImagePyramidCuda<ushort, 1, 3>;
template
class ImagePyramidCuda<uchar, 4, 3>;
template
class ImagePyramidCuda<uchar, 3, 3>;
template
class ImagePyramidCuda<uchar, 1, 3>;

template
class ImagePyramidCuda<float, 4, 3>;
template
class ImagePyramidCuda<float, 3, 3>;
template
class ImagePyramidCuda<float, 1, 3>;

template
class RGBDImagePyramidCuda<3>;
} // cuda
} // open3d