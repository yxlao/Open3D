//
// Created by wei on 10/2/18.
//

#include <Open3D/Open3D.h>
#include <Cuda/Odometry/Reduction2DCuda.h>
#include <Cuda/Geometry/ImageCuda.h>
#include <opencv2/opencv.hpp>
#include <gtest/gtest.h>

const std::string kBasePath = "../../../examples/TestData/";

using namespace open3d;
using namespace open3d::cuda;
using namespace open3d::utility;
using namespace open3d::io;

TEST(ReductionCuda, SumInt) {
    using namespace open3d;

    cv::Mat im = cv::imread(kBasePath + "RGBD/other_formats/TUM_color.png",
                            cv::IMREAD_UNCHANGED);
    cv::cvtColor(im, im, cv::COLOR_BGR2GRAY);
    im.convertTo(im, CV_32SC1);
    ImageCuda<int, 1> im_cuda;
    im_cuda.Upload(im);

    {
        Timer timer;
        int sum_cpu = 0;
        for (int i = 0; i < im.rows; ++i) {
            for (int j = 0; j < im.cols; ++j) {
                sum_cpu += im.at<int>(i, j);
            }
        }

        /** Mark harris old-fashioned code (used in InfiniTAM) **/
        float time_v1 = 0;
        const int test_cases = 10000;
        for (int i = 0; i < test_cases; ++i) {
            timer.Start();
            int sum = ReduceSum2D<int, 1>(im_cuda);
            timer.Stop();

            EXPECT_EQ(sum_cpu * TEST_ARRAY_SIZE, sum);
            time_v1 += timer.GetDuration();
        }
        time_v1 /= test_cases;

        /** __shfl_down code in ElasticFusion **/
        float time_v2 = 0;
        for (int i = 0; i < test_cases; ++i) {
            timer.Start();
            int sum = ReduceSum2DShuffle<int, 1>(im_cuda);
            timer.Stop();
            EXPECT_EQ(sum_cpu * TEST_ARRAY_SIZE, sum);
            time_v2 += timer.GetDuration();
        }
        time_v2 /= test_cases;

        /** naive atomicAdd **/
        float time_v3 = 0;
        for (int i = 0; i < test_cases; ++i) {
            timer.Start();
            int sum = AtomicSum<int, 1>(im_cuda);
            timer.Stop();
            EXPECT_EQ(sum_cpu * TEST_ARRAY_SIZE, sum);
            time_v3 += timer.GetDuration();
        }
        time_v3 /= test_cases;

        PrintInfo(">>> Average running time (ms) \n"
                  "> InfiniTAM: %.4f\n"
                  "> ElasticFusion: %.4f\n"
                  "> atomicAdd: %.4f\n",
                  time_v1, time_v2, time_v3);
    }
}

TEST(ReductionCuda, SumFloat) {
    using namespace open3d;

    cv::Mat im = cv::imread(kBasePath + "RGBD/other_formats/TUM_depth.png",
                            cv::IMREAD_UNCHANGED);
    ImageCuda<ushort, 1> im_cuda;
    ImageCuda<float, 1> imf_cuda;
    im_cuda.Upload(im);
    imf_cuda = im_cuda.ConvertToFloat();
    cv::Mat imf = imf_cuda.DownloadMat();

    for (int i = 0; i < im.rows; ++i) {
        for (int j = 0; j < im.cols; ++j) {
            EXPECT_NEAR(imf.at<float>(i, j), im.at<unsigned short>(i, j), 1);
        }
    }

    {
        Timer timer;
        float sum_cpu = 0;
        for (int i = 0; i < im.rows; ++i) {
            for (int j = 0; j < im.cols; ++j) {
                sum_cpu += imf.at<float>(i, j);
            }
        }

        const float kEpsilon = 1e-3f;
        const float kFactor = 5000.0f;
        const float kPixelNumbers = imf_cuda.width_ * imf_cuda.height_;

        float time_v1 = 0;
        const int test_cases = 10000;
        for (int i = 0; i < test_cases; ++i) {
            timer.Start();
            float sum = ReduceSum2D<float, 1>(imf_cuda);
            timer.Stop();
            EXPECT_NEAR(sum / (TEST_ARRAY_SIZE), sum_cpu,
                        kPixelNumbers * kFactor * kEpsilon);
            time_v1 += timer.GetDuration();
        }
        time_v1 /= test_cases;

        float time_v2 = 0;
        for (int i = 0; i < test_cases; ++i) {
            timer.Start();
            float sum = ReduceSum2DShuffle<float, 1>(imf_cuda);
            timer.Stop();
            EXPECT_NEAR(sum / (TEST_ARRAY_SIZE), sum_cpu,
                        kPixelNumbers * kFactor * kEpsilon);
            time_v2 += timer.GetDuration();
        }
        time_v2 /= test_cases;

        float time_v3 = 0;
        for (int i = 0; i < test_cases; ++i) {
            timer.Start();
            float sum = AtomicSum<float, 1>(imf_cuda);
            timer.Stop();
            EXPECT_NEAR(sum / (TEST_ARRAY_SIZE), sum_cpu,
                        kPixelNumbers * kFactor * kEpsilon);
            time_v3 += timer.GetDuration();
        }
        time_v3 /= test_cases;

        PrintInfo(">>> Average running time (ms) \n"
                  "> InfiniTAM: %.4f\n"
                  "> ElasticFusion: %.4f\n"
                  "> atomicAdd: %.4f\n",
                  time_v1, time_v2, time_v3);
    }
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    RUN_ALL_TESTS();
}