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
	/** (perhaps) We don't need reference count here,
	 * as long as ImageCudaServer are be handled properly ... **/
private:
	ImageCudaServer<T> images_[N];

public:
	__HOSTDEVICE__ ImageCudaServer<T>& get(size_t level) {
		assert(level < N);
		return images_[level];
	}

	friend class ImagePyramidCuda<T, N>;
};

template<typename T, size_t N>
class ImagePyramidCuda {
private:
	ImagePyramidCudaServer<T, N> server_;

private:
	ImageCuda<T> images_[N];

public:
	ImagePyramidCuda() {}
	~ImagePyramidCuda() { Release(); }

	void Create(int width, int height);
	void Release();

	void Build(const ImageCuda<T> &image);
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

	const ImagePyramidCudaServer<T, N>& server() const {
		return server_;
	}
	ImagePyramidCudaServer<T, N>& server() {
		return server_;
	}
};
}

#endif //OPEN3D_IMAGEPYRAMIDCUDA_H
