//
// Created by wei on 10/2/18.
//

#include <Cuda/Odometry/Reduction2DCuda.h>
#include <Cuda/Geometry/ImageCuda.h>
#include <Core/Core.h>
#include <opencv2/opencv.hpp>

int main() {
	using namespace three;

	cv::Mat im = cv::imread(
		//"../../../Test/TestData/RGBD/other_formats/TUM_depth.png",
		"../../../Test/TestData/lena_gray.jpg",
		cv::IMREAD_UNCHANGED);

	ImageCuda<Vector1b> im_cuda;
	ImageCuda<Vector1f> imf_cuda;
	im_cuda.Upload(im);
	imf_cuda = im_cuda.ToFloat();

	/**
	 * Test int (for correctness)
	 */
	{
		Timer timer;
		timer.Start();
		int sum_mat = 0;
		for (int i = 0; i < im.rows; ++i) {
			for (int j = 0; j < im.cols; ++j) {
				sum_mat += im.at<uchar>(i, j);
			}
		}
		timer.Stop();
		PrintInfo("CPU version: %.4f milliseconds\n", timer.GetDuration());

		float time_v1 = 0;
		const int test_cases = 10000;
		for (int i = 0; i < test_cases; ++i) {
			timer.Start();
			int sum = ReduceSum2D<Vector1b, int>(im_cuda);
			timer.Stop();
			assert(sum_mat * TEST_ARRAY_SIZE == sum);
			time_v1 += timer.GetDuration();
		}
		time_v1 /= test_cases;

		float time_v2 = 0;
		for (int i = 0; i < test_cases; ++i) {
			timer.Start();
			int sum = ReduceSum2DShuffle<Vector1b, int>(im_cuda);
			timer.Stop();
			assert(sum_mat * TEST_ARRAY_SIZE == sum);
			time_v2 += timer.GetDuration();
		}
		time_v2 /= test_cases;

		float time_v3 = 0;
		for (int i = 0; i < test_cases; ++i) {
			timer.Start();
			int sum = AtomicSum<Vector1b, int>(im_cuda);
			timer.Stop();
			assert(sum_mat * TEST_ARRAY_SIZE == sum);
			time_v3 += timer.GetDuration();
		}
		time_v3 /= test_cases;

		PrintInfo("CUDA version v1: %.4f v2: %.4f v3: %.4f\n",
				  time_v1, time_v2, time_v3);
	}

	/** Test float (for speed)
	  * Floating point sum is not guaranteed to be identical
	  **/
	{
		Timer timer;
		timer.Start();
		float sum_mat = 0;
		for (int i = 0; i < im.rows; ++i) {
			for (int j = 0; j < im.cols; ++j) {
				sum_mat += im.at<uchar>(i, j);
			}
		}
		timer.Stop();
		PrintInfo("CPU version: %.4f milliseconds\n", timer.GetDuration());

		float time_v1 = 0;
		const int test_cases = 10000;
		for (int i = 0; i < test_cases; ++i) {
			timer.Start();
			float sum = ReduceSum2D<Vector1f, float>(imf_cuda);
			timer.Stop();
			if (i == 0)
				PrintInfo("Diff: %f\n",
						  (sum / TEST_ARRAY_SIZE - sum_mat) /
							  (imf_cuda.width() * imf_cuda.height()));
			time_v1 += timer.GetDuration();
		}
		time_v1 /= test_cases;

		float time_v2 = 0;
		for (int i = 0; i < test_cases; ++i) {
			timer.Start();
			float sum = ReduceSum2DShuffle<Vector1f, float>(imf_cuda);
			timer.Stop();
			if (i == 0)
				PrintInfo("Diff: %f\n",
						  (sum / TEST_ARRAY_SIZE - sum_mat) /
							  (imf_cuda.width() * imf_cuda.height()));
			time_v2 += timer.GetDuration();
		}
		time_v2 /= test_cases;

		float time_v3 = 0;
		for (int i = 0; i < test_cases; ++i) {
			timer.Start();
			float sum = AtomicSum<Vector1f, float>(imf_cuda);
			timer.Stop();
			if (i == 0)
				PrintInfo("Diff: %f\n",
						  (sum / TEST_ARRAY_SIZE - sum_mat) /
							  (imf_cuda.width() * imf_cuda.height()));
			time_v3 += timer.GetDuration();
		}
		time_v3 /= test_cases;

		PrintInfo("CUDA version v1: %.4f v2: %.4f v3: %.4f\n",
				  time_v1, time_v2, time_v3);
	}
}