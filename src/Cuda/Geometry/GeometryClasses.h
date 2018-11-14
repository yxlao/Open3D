//
// Created by wei on 9/27/18.
//

#ifndef OPEN3D_GEOMETRY_H
#define OPEN3D_GEOMETRY_H

#pragma once

#include <cstdlib>

namespace open3d {

template<typename T>
class ImageCudaServer;
template<typename T>
class ImageCuda;

template<typename T, size_t N>
class ImagePyramidCudaServer;
template<typename T, size_t N>
class ImagePyramidCuda;

class RGBDImageCudaServer;
class RGBDImageCuda;

template<size_t N>
class RGBDImagePyramidCudaServer;
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

class TriangleMeshCudaServer;
class TriangleMeshCuda;

class PointCloudCudaServer;
class PointCloudCuda;


}
#endif //OPEN3D_GEOMETRY_H
