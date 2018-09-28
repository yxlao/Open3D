//
// Created by wei on 9/27/18.
//

#include <Core/Core.h>
#include <Cuda/Geometry/ImageCuda.h>
#include <Cuda/Geometry/Vector.h>
#include <opencv2/opencv.hpp>

template<typename T>

void CheckUploadAndDownloadConsistency(std::string path) {
	using namespace three;
	cv::Mat image = cv::imread(path, cv::IMREAD_UNCHANGED);
	cv::imshow("raw", image);
	cv::waitKey(-1);

	ImageCuda<T> image_cuda, image_cuda_copy;
	PrintInfo("Uploading ...\n");
	image_cuda.Upload(image);

	PrintInfo("Copying ...\n");
	image_cuda.CopyTo(image_cuda_copy);

	PrintInfo("Downloading ...\n");
	cv::Mat downloaded_image = image_cuda.Download();
	cv::Mat downloaded_image_copy = image_cuda_copy.Download();

	for (int i = 0; i < image.rows; ++i) {
		for (int j = 0; j < image.cols; ++j) {
			assert(image.at<short>(i, j)
			    == downloaded_image.at<short>(i, j));
			assert(image.at<short>(i, j)
			    == downloaded_image_copy.at<short>(i, j));
		}
	}
	PrintInfo("Consistency check passed.\n");

	image_cuda.Destroy();
	image_cuda_copy.Destroy();
}

template<typename T>
void CheckDownsampling(std::string path) {
	using namespace three;
	cv::Mat image = cv::imread(path, cv::IMREAD_UNCHANGED);

	ImageCuda<T> image_cuda, image_cuda_low;
	PrintInfo("Uploading ...\n");
	image_cuda.Upload(image);

	PrintInfo("Downsampling ...\n");
	image_cuda_low = image_cuda.Downsample();

	PrintInfo("Downloading ...\n");
	cv::Mat downloaded = image_cuda_low.Download();
	cv::imshow("downsampled", downloaded);
	cv::waitKey(-1);
}

int main(int argc, char** argv) {
	using namespace three;

	PrintInfo("#1 Checking depth.\n");
	CheckUploadAndDownloadConsistency<Vector1s>(
		"../../../Test/TestData/RGBD/other_formats/TUM_depth.png");
	CheckDownsampling<Vector1s>(
		"../../../Test/TestData/RGBD/other_formats/TUM_depth.png");

	PrintInfo("#2 Checking color.\n");
	CheckUploadAndDownloadConsistency<Vector3b>(
		"../../../Test/TestData/RGBD/other_formats/TUM_color.png");
	CheckDownsampling<Vector3b>(
		"../../../Test/TestData/RGBD/other_formats/TUM_color.png");

	return 0;
}