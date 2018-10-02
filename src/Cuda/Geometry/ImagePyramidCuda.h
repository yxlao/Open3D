//
// Created by wei on 9/27/18.
//

#ifndef OPEN3D_IMAGEPYRAMIDCUDA_H
#define OPEN3D_IMAGEPYRAMIDCUDA_H

#include "GeometryClasses.h"
#include "ImageCuda.h"

namespace three {
template<typename VecType, size_t N>
class ImagePyramidCudaServer {
	/** (perhaps) We don't need reference count here,
	 * as long as ImageCudaServer are be handled properly ... **/
private:
	ImageCudaServer<VecType> images_[N];

public:
	__HOSTDEVICE__ ImageCudaServer<VecType>& get(size_t level) {
		assert(level < N);
		return images_[level];
	}

	friend class ImagePyramidCuda<VecType, N>;
};

template<typename VecType, size_t N>
class ImagePyramidCuda {
private:
	ImagePyramidCudaServer<VecType, N> server_;

private:
	ImageCuda<VecType> images_[N];

public:
	ImagePyramidCuda() {}
	~ImagePyramidCuda() { Release(); }

	void Create(int width, int height);
	void Release();

	void Build(const ImageCuda<VecType> &image);
	void Build(cv::Mat &m);
	std::vector<cv::Mat> Download();

	int width(int level = 0) const {
		return images_[level].width();
	}
	int height(int level = 0) const {
		return images_[level].height();
	}
	int pitch(int level = 0) const {
		return images_[level].pitch();
	}

	const ImagePyramidCudaServer<VecType, N>& server() const {
		return server_;
	}
	ImagePyramidCudaServer<VecType, N>& server() {
		return server_;
	}
};
}

#endif //OPEN3D_IMAGEPYRAMIDCUDA_H
