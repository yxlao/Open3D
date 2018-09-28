//
// Created by wei on 9/27/18.
//

#ifndef OPEN3D_IMAGEPYRAMIDCUDA_H
#define OPEN3D_IMAGEPYRAMIDCUDA_H

#include "GeometryClasses.h"
#include "ImageCuda.h"

namespace three {
template<typename T, size_t N>
class ImagePyramidCudaServer {
private:
	ImageCudaServer<T> images_[N];

public:
	friend class ImagePyramidCuda<T, N>;
};

template<typename T, size_t N>
class ImagePyramidCuda {
private:
	ImagePyramidCuda<T, N> server_;

public:

};
}

#endif //OPEN3D_IMAGEPYRAMIDCUDA_H
