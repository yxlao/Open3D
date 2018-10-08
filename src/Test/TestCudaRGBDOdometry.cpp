//
// Created by wei on 10/6/18.
//

#include <Cuda/Odometry/RGBDOdometryCuda.h>
#include <string>
#include <vector>
#include <Core/Core.h>
#include <iostream>
#include <opencv2/opencv.hpp>

int main() {
	using namespace three;

	std::string base_path = "../../../Test/TestData/RGBD/";

	cv::Mat source_color = cv::imread(base_path + "color/00001.jpg");
	cv::cvtColor(source_color, source_color, cv::COLOR_BGR2GRAY);
	source_color.convertTo(source_color, CV_32FC1, 1.0f / 255.0f);

	cv::Mat target_color = cv::imread(base_path + "color/00000.jpg");
	cv::cvtColor(target_color, target_color, cv::COLOR_BGR2GRAY);
	target_color.convertTo(target_color, CV_32FC1, 1.0f / 255.0f);

	cv::Mat source_depth = cv::imread(base_path + "depth/00001.png",
									  cv::IMREAD_UNCHANGED);
	source_depth.convertTo(source_depth, CV_32FC1, 0.001f);
	cv::Mat target_depth = cv::imread(base_path + "depth/00000.png",
									  cv::IMREAD_UNCHANGED);
	target_depth.convertTo(target_depth, CV_32FC1, 0.001f);

	ImageCuda<Vector1f> source_I, target_I, source_D, target_D;
	source_I.Upload(source_color);
	target_I.Upload(target_color);
	source_D.Upload(source_depth);
	target_D.Upload(target_depth);

	RGBDOdometryCuda<3> odometry;
	odometry.server()->pinhole_camera_intrinsics_.Set(
		640, 480, 525.0, 525.0, 319.5, 239.5);
	odometry.transform_source_to_target_ = RGBDOdometryCuda<3>::Matrix4f::Identity();
	odometry.SetParameters(0.2f, 0.1f, 4.0f, 0.07f);
	odometry.Apply(source_D, source_I, target_D, target_I);
	return 0;
}