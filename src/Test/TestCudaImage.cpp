//
// Created by wei on 9/27/18.
//

#include <Core/Core.h>
#include <Cuda/Geometry/ImageCuda.h>
#include <Cuda/Geometry/ImagePyramidCuda.h>
#include <Cuda/Geometry/Vector.h>
#include <opencv2/opencv.hpp>

template<typename T>
void CheckUploadAndDownloadConsistency(std::string path) {
	using namespace three;
	cv::Mat image = cv::imread(path, cv::IMREAD_UNCHANGED);
	cv::imshow("raw", image);
	cv::waitKey(-1);

	Timer timer;
	ImageCuda<T> image_cuda, image_cuda_copy;

	timer.Start();
	image_cuda.Upload(image);
	timer.Stop();
	PrintInfo("Upload finished in %.3f milliseconds...\n", timer.GetDuration());

	timer.Start();
	image_cuda.CopyTo(image_cuda_copy);
	timer.Stop();
	PrintInfo("Copy finished in %.3f milliseconds...\n", timer.GetDuration());
	

	timer.Start();
	cv::Mat downloaded_image = image_cuda.Download();
	timer.Stop();
	PrintInfo("Download finished in %.3f milliseconds...\n", timer.GetDuration());
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
}

template<typename T>
void CheckDownsampling(std::string path) {
	using namespace three;
	cv::Mat image = cv::imread(path, cv::IMREAD_UNCHANGED);

	Timer timer;
	ImageCuda<T> image_cuda, image_cuda_low;

	timer.Start();
	image_cuda.Upload(image);
	timer.Stop();
	PrintInfo("Upload finished in %.3f milliseconds...\n", timer.GetDuration());

	timer.Start();
	image_cuda_low = image_cuda.Downsample();
	timer.Stop();
	PrintInfo("Downsample finished in %.3f milliseconds...\n", timer.GetDuration());

	timer.Start();
	cv::Mat downloaded = image_cuda_low.Download();
	timer.Stop();
	PrintInfo("Download finished in %.3f milliseconds...\n", timer.GetDuration());

	cv::imshow("downsampled", downloaded);
	cv::waitKey(-1);
	cv::destroyAllWindows();
}

template<typename T>
void CheckGaussian(std::string path) {
	using namespace three;
	cv::Mat image = cv::imread(path, cv::IMREAD_UNCHANGED);

	Timer timer;
	ImageCuda<T> image_cuda;

	timer.Start();
	image_cuda.Upload(image);
	timer.Stop();
	PrintInfo("Upload finished in %.3f milliseconds...\n", timer.GetDuration());

	timer.Start();
	ImageCuda<T> image_cuda_blurred = image_cuda.Gaussian(Gaussian3x3);
	timer.Stop();
	PrintInfo("Gaussian3x3 finished in %.3f milliseconds...\n", timer.GetDuration());

	cv::Mat downloaded = image_cuda_blurred.Download();
	cv::imshow("Gaussian3x3", downloaded);
	cv::waitKey(-1);

	timer.Start();
	image_cuda_blurred = image_cuda.Gaussian(Gaussian5x5);
	timer.Stop();
	PrintInfo("Gaussian5x5 finished in %.3f milliseconds...\n", timer.GetDuration());

	downloaded = image_cuda_blurred.Download();
	cv::imshow("Gaussian5x5", downloaded);
	cv::waitKey(-1);

	timer.Start();
	image_cuda_blurred = image_cuda.Gaussian(Gaussian7x7);
	timer.Stop();
	PrintInfo("Gaussian7x7 finished in %.3f milliseconds...\n", timer.GetDuration());

	downloaded = image_cuda_blurred.Download();
	cv::imshow("Gaussian7x7", downloaded);
	cv::waitKey(-1);
	cv::destroyAllWindows();
}


template<typename T>
void CheckBilateral(std::string path) {
	using namespace three;
	cv::Mat image = cv::imread(path, cv::IMREAD_UNCHANGED);

	Timer timer;
	ImageCuda<T> image_cuda, filtered_image_cuda;

	timer.Start();
	image_cuda.Upload(image);
	timer.Stop();
	PrintInfo("Upload finished in %.3f milliseconds...\n", timer.GetDuration());

	float val_sigma = 20;
	timer.Start();
	image_cuda.Bilateral(filtered_image_cuda, Gaussian5x5, val_sigma);
	timer.Stop();

	cv::Mat downloaded = filtered_image_cuda.Download();
	PrintInfo("Sigma: %.3f in  %.3f milliseconds\n",
			  val_sigma, timer.GetDuration());
	cv::imshow("Bilateral", downloaded);
	cv::waitKey(-1);
}

template<typename T>
void CheckToFloatConversion(std::string path, float scale, float offset) {
	using namespace three;
	cv::Mat image = cv::imread(path, cv::IMREAD_UNCHANGED);

	Timer timer;
	ImageCuda<T> image_cuda;
	ImageCuda<typename T::VecTypef> imagef_cuda;
	timer.Start();
	image_cuda.Upload(image);
	timer.Stop();
	PrintInfo("Upload finished in %.3f milliseconds...\n", timer.GetDuration());

	timer.Start();
	imagef_cuda = image_cuda.ToFloat(scale, offset);
	timer.Stop();
	PrintInfo("Conversion finished in %.3f milliseconds...\n", timer.GetDuration());

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
	cv::destroyAllWindows();
}

template<typename T>
void CheckGradient(std::string path) {
	using namespace three;
	cv::Mat image = cv::imread(path, cv::IMREAD_UNCHANGED);

	Timer timer;
	ImageCuda<T> image_cuda;
	timer.Start();
	image_cuda.Upload(image);
	timer.Stop();
	PrintInfo("Upload finished in %.3f milliseconds...\n", timer.GetDuration());

	timer.Start();
	auto gradients = image_cuda.Sobel();
	timer.Stop();
	PrintInfo("Gradient finished in %.3f milliseconds...\n", timer.GetDuration());
	auto dx = std::get<0>(gradients);
	auto dy = std::get<1>(gradients);

	cv::Mat downloaded_dx = dx.Download();
	cv::Mat downloaded_dy = dy.Download();
	cv::imshow("dx", downloaded_dx / 255.0f);
	cv::imshow("dy", downloaded_dy / 255.0f);
	cv::waitKey(-1);
	cv::destroyAllWindows();
}

template<typename T, size_t N>
void CheckPyramid(std::string path) {
	using namespace three;
	cv::Mat image = cv::imread(path, cv::IMREAD_UNCHANGED);

	Timer timer;
	ImagePyramidCuda<T, N> pyramid;
	timer.Start();
	pyramid.Build(image);
	timer.Stop();
	PrintInfo("> pass 1 finished in %.3f milliseconds...\n", timer.GetDuration());

	/* Test memory-use */
	timer.Start();
	pyramid.Build(image);
	timer.Stop();
	PrintInfo("> pass 2 finished in %.3f milliseconds...\n", timer.GetDuration());

	timer.Start();
	pyramid.Build(image);
	timer.Stop();
	PrintInfo("> pass 3 finished in %.3f milliseconds...\n", timer.GetDuration());

	std::vector<cv::Mat> downloaded_images = pyramid.Download();
	std::stringstream ss;
	for (int level = 0; level < N; ++level) {
		ss.str("");
		ss << "level-" << level;
		cv::imshow(ss.str(), downloaded_images[level]);
	}
	cv::waitKey(-1);
	cv::destroyAllWindows();
}

int main(int argc, char** argv) {
	using namespace three;

	PrintInfo("#1 Checking depth.\n");
	std::string depth_path =
		"../../../Test/TestData/RGBD/other_formats/TUM_depth.png";

	CheckUploadAndDownloadConsistency<Vector1s>(depth_path);
	PrintInfo("------\n");
	CheckDownsampling<Vector1s>(depth_path);
	PrintInfo("------\n");
	CheckToFloatConversion<Vector1s>(depth_path, 1.0f / 5000.0f, 0.0f);
	PrintInfo("------\n");
	CheckGradient<Vector1s>(depth_path);
	PrintInfo("------\n");
	CheckGaussian<Vector1s>(depth_path);
	PrintInfo("------\n");

	PrintInfo("#2 Checking color.\n");
	std::string color_path =
		"../../../Test/TestData/RGBD/other_formats/TUM_color.png";

	CheckUploadAndDownloadConsistency<Vector3b>(color_path);
	PrintInfo("------\n");
	CheckDownsampling<Vector3b>(color_path);
	PrintInfo("------\n");
	CheckToFloatConversion<Vector3b>(color_path, 1.0f / 255.0f, 0.0f);
	PrintInfo("------\n");
	CheckGradient<Vector3b>(color_path);
	PrintInfo("------\n");
	CheckGaussian<Vector3b>(color_path);
	PrintInfo("------\n");

	PrintInfo("#3 Checking grayscale.\n");
	std::string grayscale_path = "../../../Test/TestData/lena_gray.jpg";
	cv::Mat a = cv::imread(grayscale_path);
	cv::cvtColor(a, a, cv::COLOR_BGR2GRAY);
	cv::imshow("show", a);
	cv::waitKey(-1);
	std::cout << a.type() << std::endl;
	CheckDownsampling<Vector1b>(grayscale_path);
	PrintInfo("------\n");
	CheckGradient<Vector1b>(grayscale_path);
	PrintInfo("------\n");
	CheckGaussian<Vector1b>(grayscale_path);
	PrintInfo("------\n");

	PrintInfo("#4 Checking ImagePyramid.\n");
	CheckPyramid<Vector1s, 4>(depth_path);
	PrintInfo("------\n");
	CheckPyramid<Vector3b, 4>(color_path);
	PrintInfo("------\n");
	CheckPyramid<Vector1b, 4>(grayscale_path);
	PrintInfo("------\n");

	PrintInfo("#4 Checking Bilateral.\n");
	CheckBilateral<Vector1s>(depth_path);
	PrintInfo("------\n");
	CheckBilateral<Vector3b>(color_path);
	PrintInfo("------\n");
	CheckBilateral<Vector1b>(grayscale_path);

	return 0;
}