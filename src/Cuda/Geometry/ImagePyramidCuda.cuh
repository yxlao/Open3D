//
// Created by wei on 9/27/18.
//

#ifndef OPEN3D_IMAGEPYRAMIDCUDA_CUH
#define OPEN3D_IMAGEPYRAMIDCUDA_CUH

#include "ImagePyramidCuda.h"
#include <Cuda/Common/UtilsCuda.h>
#include <Core/Core.h>

namespace three {
template<typename T, size_t N>
void ImagePyramidCuda<T, N>::Create(int width, int height) {

	for (size_t level = 0; level < N; ++level) {
		int w = width >> level;
		int h = height >> level;
		if (w == 0 || h == 0) {
			PrintError("Invalid width %d || height %d at level %d!\n",
				w, h, level);
			return;
		}

		images_[level].Create(width, height);
		server_.get(level) = images_[level].server();
	}
}

template<typename T, size_t N>
void ImagePyramidCuda<T, N>::Release() {
	for (size_t level = 0; level < N; ++level) {
		images_[level].Release();
	}
}

template<typename VecType, size_t N>
void ImagePyramidCuda<VecType, N>::Build(cv::Mat &m) {
	images_[0].Upload(m);
	for (size_t level = 1; level < N; ++level) {
		images_[level - 1].Downsample(images_[level]);
	}
}

template<typename VecType, size_t N>
void ImagePyramidCuda<VecType, N>::Build(const ImageCuda<VecType> &image) {
	image.CopyTo(images_[0]);
	for (size_t level = 1; level < N; ++level) {
		images_[level - 1].Downsample(images_[level]);
	}
}

template<typename VecType, size_t N>
std::vector<cv::Mat> ImagePyramidCuda<VecType, N>::Download() {
	std::vector<cv::Mat> result;
	for (size_t level = 0; level < N; ++level) {
		result.emplace_back(images_[level].Download());
	}
	return result;
}

}
#endif //OPEN3D_IMAGEPYRAMIDCUDA_CUH
