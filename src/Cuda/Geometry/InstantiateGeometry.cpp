//
// Created by wei on 11/9/18.
//

#include "ImageCudaHost.hpp"
#include "RGBDImageCudaHost.hpp"

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

} // cuda
} // open3d