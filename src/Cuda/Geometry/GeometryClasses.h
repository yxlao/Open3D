//
// Created by wei on 9/27/18.
//

#ifndef OPEN3D_GEOMETRY_H
#define OPEN3D_GEOMETRY_H

#pragma once

#include <cstdlib>

namespace open3d {

namespace cuda {
template<typename T>
class ImageCudaDevice;
template<typename T>
class ImageCuda;

template<typename T, size_t N>
class ImagePyramidCudaDevice;
template<typename T, size_t N>
class ImagePyramidCuda;

class RGBDImageCudaDevice;
class RGBDImageCuda;

template<size_t N>
class RGBDImagePyramidCudaDevice;
template<size_t N>
class RGBDImagePyramidCuda;

enum VertexType {
    VertexRaw = 0,
    VertexWithNormal = 1,
    VertexWithColor = (1 << 1),
    VertexWithNormalAndColor = VertexWithNormal | VertexWithColor,
    VertexAsSurfel = (1 << 2),
    VertexTypeUnknown = (1 << 30)
};

class TriangleMeshCudaDevice;
class TriangleMeshCuda;

class PointCloudCudaDevice;
class PointCloudCuda;

} // cuda
} // open3d
#endif //OPEN3D_GEOMETRY_H
