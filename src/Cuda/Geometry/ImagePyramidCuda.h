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
private:
	/** Unfortunately, we cannot use shared_ptr for CUDA data structures **/
	/** We even cannot use CPU pointers **/
	/** -- We have to call ConnectSubServers() to explicitly link them. **/
	ImageCudaServer<VecType> images_[N];

public:
	__HOSTDEVICE__ ImageCudaServer<VecType>& level(size_t i) {
		assert(i < N);
		return images_[i];
	}

	friend class ImagePyramidCuda<VecType, N>;
};

template<typename VecType, size_t N>
class ImagePyramidCuda {
private:
	std::shared_ptr<ImagePyramidCudaServer<VecType, N>> server_ = nullptr;

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

	void ConnectSubServers();

	ImageCuda<VecType> & level(size_t i) {
		assert(i < N);
		return images_[i];
	}
	int width(size_t i = 0) const {
		assert(i < N);
		return images_[i].width();
	}
	int height(size_t i = 0) const {
		assert(i < N);
		return images_[i].height();
	}
	int pitch(size_t i = 0) const {
		assert(i < N);
		return images_[i].pitch();
	}

	const std::shared_ptr<ImagePyramidCudaServer<VecType, N>>& server() const {
		return server_;
	}
	std::shared_ptr<ImagePyramidCudaServer<VecType, N>>& server() {
		return server_;
	}
};
}

#endif //OPEN3D_IMAGEPYRAMIDCUDA_H
