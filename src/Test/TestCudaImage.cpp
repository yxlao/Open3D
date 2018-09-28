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
			if (image.type() == CV_8UC3) {
				assert(image.at<cv::Vec3b>(i, j)
						   == downloaded_image.at<cv::Vec3b>(i, j));
				assert(image.at<cv::Vec3b>(i, j)
						   == downloaded_image_copy.at<cv::Vec3b>(i, j));
			} else if (image.type() == CV_16UC1) {
				assert(image.at<short>(i, j)
						   == downloaded_image.at<short>(i, j));
				assert(image.at<short>(i, j)
						   == downloaded_image_copy.at<short>(i, j));
			} else {
				PrintInfo("Unsupported image type %d\n", image.type());
			}
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

	image_cuda.Destroy();
	image_cuda_low.Destroy();
}

template<typename T>
void CheckGaussian(std::string path) {
	using namespace three;
	cv::Mat image = cv::imread(path, cv::IMREAD_UNCHANGED);

	ImageCuda<T> image_cuda, image_cuda_blurred;
	PrintInfo("Uploading ...\n");
	image_cuda.Upload(image);

	PrintInfo("Gaussian ...\n");
	image_cuda_blurred = image_cuda.Gaussian(Gaussian3x3);

	PrintInfo("Downloading ...\n");
	cv::Mat downloaded = image_cuda_blurred.Download();
	cv::imshow("Gaussian3x3", downloaded);
	cv::waitKey(-1);
	image_cuda_blurred.Destroy();

	PrintInfo("Gaussian ...\n");
	image_cuda_blurred = image_cuda.Gaussian(Gaussian5x5);
	PrintInfo("Downloading ...\n");
	downloaded = image_cuda_blurred.Download();
	cv::imshow("Gaussian5x5", downloaded);
	cv::waitKey(-1);
	image_cuda_blurred.Destroy();

	PrintInfo("Gaussian ...\n");
	image_cuda_blurred = image_cuda.Gaussian(Gaussian7x7);
	PrintInfo("Downloading ...\n");
	downloaded = image_cuda_blurred.Download();
	cv::imshow("Gaussian7x7", downloaded);
	cv::waitKey(-1);
	image_cuda_blurred.Destroy();

	image_cuda.Destroy();

}

template<typename T>
void CheckToFloatConversion(std::string path, float scale, float offset) {
	using namespace three;
	cv::Mat image = cv::imread(path, cv::IMREAD_UNCHANGED);

	ImageCuda<T> image_cuda;
	ImageCuda<typename T::VecTypef> imagef_cuda;
	PrintInfo("Uploading ...\n");
	image_cuda.Upload(image);

	PrintInfo("Converting ...\n");
	imagef_cuda = image_cuda.ToFloat(scale, offset);

	PrintInfo("Downloading ...\n");
	cv::Mat downloaded = imagef_cuda.Download();
	for (int i = 0; i < image.rows; ++i) {
		for (int j = 0; j < image.cols; ++j) {
			if (image.type() == CV_8UC3) {
				cv::Vec3b raw = image.at<cv::Vec3b>(i, j);
				cv::Vec3f converted = downloaded.at<cv::Vec3f>(i, j);
				cv::Vec3f diff = cv::Vec3f(
					raw[0] * scale + offset - converted[0],
					raw[1] * scale + offset - converted[1],
					raw[2] * scale + offset - converted[2]);
				assert(fabsf(diff[0]) < 1e-5 && fabsf(diff[1]) < 1e-5
				&& fabsf(diff[2]) < 1e-5);
			} else if (image.type() == CV_16UC1) {
				short raw = image.at<short>(i, j);
				float converted = downloaded.at<float>(i, j);
				float diff = raw * scale + offset - converted;
				assert(fabsf(diff) < 1e-5);
			} else {
				PrintInfo("Unsupported image type %d\n", image.type());
			}
		}
	}

	cv::imshow("converted", downloaded);
	cv::waitKey(-1);
	image_cuda.Destroy();
	imagef_cuda.Destroy();
}

template<typename T>
void CheckGradient(std::string path) {
	using namespace three;
	cv::Mat image = cv::imread(path, cv::IMREAD_UNCHANGED);

	ImageCuda<T> image_cuda;
	PrintInfo("Uploading ...\n");
	image_cuda.Upload(image);

	PrintInfo("Computing gradient ...\n");
	auto gradients = image_cuda.Gradient();
	auto dx = std::get<0>(gradients);
	auto dy = std::get<1>(gradients);

	PrintInfo("Downloading ...\n");
	cv::Mat downloaded_dx = dx.Download();
	cv::Mat downloaded_dy = dy.Download();
	cv::imshow("dx", downloaded_dx / 255.0f);
	cv::imshow("dy", downloaded_dy / 255.0f);
	cv::waitKey(-1);

	image_cuda.Destroy();
	dx.Destroy();
	dy.Destroy();
}

int main(int argc, char** argv) {
	using namespace three;

	PrintInfo("#1 Checking depth.\n");
	CheckUploadAndDownloadConsistency<Vector1s>(
		"../../../Test/TestData/RGBD/other_formats/TUM_depth.png");
	CheckDownsampling<Vector1s>(
		"../../../Test/TestData/RGBD/other_formats/TUM_depth.png");
	CheckToFloatConversion<Vector1s>(
		"../../../Test/TestData/RGBD/other_formats/TUM_depth.png",
		1.0f / 5000.0f, 0.0f);
	CheckGradient<Vector1s>(
		"../../../Test/TestData/RGBD/other_formats/TUM_depth.png");
	CheckGaussian<Vector1s>(
		"../../../Test/TestData/RGBD/other_formats/TUM_depth.png");

	PrintInfo("#2 Checking color.\n");
	CheckUploadAndDownloadConsistency<Vector3b>(
		"../../../Test/TestData/RGBD/other_formats/TUM_color.png");
	CheckDownsampling<Vector3b>(
		"../../../Test/TestData/RGBD/other_formats/TUM_color.png");
	CheckToFloatConversion<Vector3b>(
		"../../../Test/TestData/RGBD/other_formats/TUM_color.png",
		1.0f / 255.0f, 0.0f);
	CheckGradient<Vector3b>(
		"../../../Test/TestData/RGBD/other_formats/TUM_color.png");
	CheckGaussian<Vector3b>(
		"../../../Test/TestData/RGBD/other_formats/TUM_color.png");

	CheckGradient<Vector1b>(
		"../../../Test/TestData/lena_gray.jpg");
	CheckGaussian<Vector1b>(
		"../../../Test/TestData/lena_gray.jpg");

	return 0;
}