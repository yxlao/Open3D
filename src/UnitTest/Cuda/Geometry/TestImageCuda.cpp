//
// Created by wei on 9/27/18.
//

#include <Core/Core.h>
#include <Cuda/Geometry/ImageCuda.h>
#include <Cuda/Geometry/ImagePyramidCuda.h>
#include <Cuda/Common/VectorCuda.h>
#include <opencv2/opencv.hpp>
#include "UnitTest.h"

template<typename T>
void CheckUploadAndDownloadConsistency(std::string path) {
    using namespace open3d;
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
    image_cuda_copy.CopyFrom(image_cuda);
    timer.Stop();
    PrintInfo("Copy finished in %.3f milliseconds...\n", timer.GetDuration());

    timer.Start();
    cv::Mat downloaded_image = image_cuda.DownloadMat();
    timer.Stop();
    PrintInfo("Download finished in %.3f milliseconds...\n",
              timer.GetDuration());
    cv::Mat downloaded_image_copy = image_cuda_copy.DownloadMat();

    for (int i = 0; i < image.rows; ++i) {
        for (int j = 0; j < image.cols; ++j) {
            if (image.type() == CV_8UC3) {
                EXPECT_EQ(image.at<cv::Vec3b>(i, j),
                          downloaded_image.at<cv::Vec3b>(i, j));
                EXPECT_EQ(image.at<cv::Vec3b>(i, j),
                          downloaded_image_copy.at<cv::Vec3b>(i, j));
            } else if (image.type() == CV_16UC1) {
                EXPECT_EQ(image.at<short>(i, j),
                          downloaded_image.at<short>(i, j));
                EXPECT_EQ(image.at<short>(i, j),
                          downloaded_image_copy.at<short>(i, j));
            } else if (image.type() == CV_8UC1) {
                EXPECT_EQ(image.at<uchar>(i, j),
                          downloaded_image.at<uchar>(i, j));
                EXPECT_EQ(image.at<uchar>(i, j),
                          downloaded_image_copy.at<uchar>(i, j));
            } else {
                PrintInfo("Unsupported image type %d\n", image.type());
            }
        }
    }
    PrintInfo("Consistency check passed.\n");
}

template<typename T>
void CheckDownsampling(std::string path) {
    using namespace open3d;
    cv::Mat image = cv::imread(path, cv::IMREAD_UNCHANGED);

    Timer timer;
    ImageCuda<T> image_cuda, image_cuda_low;

    timer.Start();
    image_cuda.Upload(image);
    timer.Stop();
    PrintInfo("Upload finished in %.3f milliseconds...\n", timer.GetDuration());

    timer.Start();
    image_cuda_low = image_cuda.Downsample(BoxFilterWithHoles);
    timer.Stop();
    PrintInfo("Downsample finished in %.3f milliseconds...\n",
              timer.GetDuration());

    timer.Start();
    cv::Mat downloaded = image_cuda_low.DownloadMat();
    timer.Stop();
    PrintInfo("Download finished in %.3f milliseconds...\n",
              timer.GetDuration());

    cv::imshow("downsampled", downloaded);
    cv::waitKey(-1);
    cv::destroyAllWindows();
}

template<typename T>
void CheckGaussian(std::string path) {
    using namespace open3d;
    cv::Mat image = cv::imread(path, cv::IMREAD_UNCHANGED);

    Timer timer;
    ImageCuda<T> image_cuda;

    timer.Start();
    image_cuda.Upload(image);
    timer.Stop();
    PrintInfo("Upload finished in %.3f milliseconds...\n", timer.GetDuration());

    timer.Start();
    ImageCuda<T> image_cuda_blurred = image_cuda.Gaussian(Gaussian3x3, false);
    timer.Stop();
    PrintInfo("Gaussian3x3 finished in %.3f milliseconds...\n",
              timer.GetDuration());

    cv::Mat downloaded = image_cuda_blurred.DownloadMat();
    cv::imshow("Gaussian3x3", downloaded);
    cv::waitKey(-1);

    timer.Start();
    image_cuda_blurred = image_cuda.Gaussian(Gaussian5x5, false);
    timer.Stop();
    PrintInfo("Gaussian5x5 finished in %.3f milliseconds...\n",
              timer.GetDuration());

    downloaded = image_cuda_blurred.DownloadMat();
    cv::imshow("Gaussian5x5", downloaded);
    cv::waitKey(-1);

    timer.Start();
    image_cuda_blurred = image_cuda.Gaussian(Gaussian7x7, false);
    timer.Stop();
    PrintInfo("Gaussian7x7 finished in %.3f milliseconds...\n",
              timer.GetDuration());

    downloaded = image_cuda_blurred.DownloadMat();
    cv::imshow("Gaussian7x7", downloaded);
    cv::waitKey(-1);
    cv::destroyAllWindows();
}

template<typename T>
void CheckBilateral(std::string path) {
    using namespace open3d;
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

    cv::Mat downloaded = filtered_image_cuda.DownloadMat();
    PrintInfo("Sigma: %.3f in  %.3f milliseconds\n",
              val_sigma, timer.GetDuration());
    cv::imshow("Bilateral", downloaded);
    cv::waitKey(-1);
}

template<typename T>
void CheckToFloatConversion(std::string path, float scale, float offset) {
    using namespace open3d;
    cv::Mat image = cv::imread(path, cv::IMREAD_UNCHANGED);

    Timer timer;
    ImageCuda<T> image_cuda;
    ImageCuda<typename T::VecTypef> imagef_cuda;
    timer.Start();
    image_cuda.Upload(image);
    timer.Stop();
    PrintInfo("Upload finished in %.3f milliseconds...\n", timer.GetDuration());

    int iter = 100;
    timer.Start();
    for (int i = 0; i < iter; ++i) {
        imagef_cuda = image_cuda.ToFloat(scale, offset);
    }
    timer.Stop();
    PrintInfo("Conversion finished in %.3f milliseconds...\n",
              timer.GetDuration() / iter);

    cv::Mat downloaded = imagef_cuda.DownloadMat();
    const float kEpsilon = 1e-5f;
    for (int i = 0; i < image.rows; ++i) {
        for (int j = 0; j < image.cols; ++j) {
            if (image.type() == CV_8UC3) {
                cv::Vec3b raw = image.at<cv::Vec3b>(i, j);
                cv::Vec3f converted = downloaded.at<cv::Vec3f>(i, j);
                EXPECT_NEAR(raw[0] * scale + offset, converted[0], kEpsilon);
                EXPECT_NEAR(raw[1] * scale + offset, converted[1], kEpsilon);
                EXPECT_NEAR(raw[2] * scale + offset, converted[2], kEpsilon);
            } else if (image.type() == CV_16UC1) {
                ushort raw = image.at<ushort>(i, j);
                float converted = downloaded.at<float>(i, j);
                EXPECT_NEAR(raw * scale + offset, converted, kEpsilon);
            } else if (image.type() == CV_8UC1) {
                uchar raw = image.at<uchar>(i, j);
                float converted = downloaded.at<float>(i, j);
                EXPECT_NEAR(raw * scale + offset, converted, kEpsilon);
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
    using namespace open3d;
    cv::Mat image = cv::imread(path, cv::IMREAD_UNCHANGED);

    Timer timer;
    ImageCuda<T> image_cuda;
    timer.Start();
    image_cuda.Upload(image);
    timer.Stop();
    PrintInfo("Upload finished in %.3f milliseconds...\n", timer.GetDuration());

    timer.Start();
    auto gradients = image_cuda.Sobel(false);
    timer.Stop();
    PrintInfo("Gradient finished in %.3f milliseconds...\n",
              timer.GetDuration());
    auto dx = std::get<0>(gradients);
    auto dy = std::get<1>(gradients);

    cv::Mat downloaded_dx = dx.DownloadMat();
    cv::Mat downloaded_dy = dy.DownloadMat();
    cv::imshow("dx", downloaded_dx / 255.0f);
    cv::imshow("dy", downloaded_dy / 255.0f);
    cv::waitKey(-1);
    cv::destroyAllWindows();
}

template<typename T>
void CheckShift(std::string path) {
    using namespace open3d;
    cv::Mat image = cv::imread(path, cv::IMREAD_UNCHANGED);

    Timer timer;
    ImageCuda<T> image_cuda;
    timer.Start();
    image_cuda.Upload(image);
    timer.Stop();
    PrintInfo("Upload finished in %.3f milliseconds...\n", timer.GetDuration());

    timer.Start();
    ImageCuda<T> shifted_image = image_cuda.Shift(-120.2f, 135.8f);
    timer.Stop();
    PrintInfo("Shifting finished in %.3f milliseconds...\n",
              timer.GetDuration());

    cv::Mat downloaded = shifted_image.DownloadMat();
    cv::imshow("shifted", downloaded);
    cv::waitKey(-1);
    cv::destroyAllWindows();
}

template<typename T, size_t N>
void CheckPyramid(std::string path) {
    using namespace open3d;
    cv::Mat image = cv::imread(path, cv::IMREAD_UNCHANGED);
    ImageCuda<T> image_cuda;
    image_cuda.Upload(image);

    Timer timer;
    ImagePyramidCuda<T, N> pyramid;
    timer.Start();
    pyramid.Build(image_cuda);
    timer.Stop();
    PrintInfo("> pass 1 finished in %.3f milliseconds...\n",
              timer.GetDuration());

    /* Test memory-use */
    timer.Start();
    pyramid.Build(image_cuda);
    timer.Stop();
    PrintInfo("> pass 2 finished in %.3f milliseconds...\n",
              timer.GetDuration());

    timer.Start();
    pyramid.Build(image_cuda);
    timer.Stop();
    PrintInfo("> pass 3 finished in %.3f milliseconds...\n",
              timer.GetDuration());

    std::vector<cv::Mat> downloaded_images = pyramid.DownloadMats();
    std::stringstream ss;
    for (int level = 0; level < N; ++level) {
        ss.str("");
        ss << "level-" << level;
        cv::imshow(ss.str(), downloaded_images[level]);
    }
    cv::waitKey(-1);
    cv::destroyAllWindows();
}

const std::string kDepthPath = "../../examples/TestData/RGBD/other_formats/TUM_depth.png";
const std::string kColorPath = "../../examples/TestData/RGBD/other_formats/TUM_color.png";
const std::string kGrayPath  = "../../examples/TestData/lena_gray.jpg";

TEST(ImageCuda, UploadAndDownload) {
    using namespace open3d;
    CheckUploadAndDownloadConsistency<Vector1s>(kDepthPath);
    CheckUploadAndDownloadConsistency<Vector3b>(kColorPath);
    CheckUploadAndDownloadConsistency<Vector1b>(kGrayPath);
}

TEST(ImageCuda, Downsampling) {
    using namespace open3d;
    CheckDownsampling<Vector1s>(kDepthPath);
    CheckDownsampling<Vector3b>(kColorPath);
    CheckDownsampling<Vector1b>(kGrayPath);
}

TEST(ImageCuda, ToFloatConversion) {
    using namespace open3d;
    CheckToFloatConversion<Vector1s>(kDepthPath, 1.0f / 5000.0f, 0.0f);
    CheckToFloatConversion<Vector3b>(kColorPath, 1.0f / 255.0f, 0.0f);
    CheckToFloatConversion<Vector1b>(kGrayPath, 1.0f / 255.0f, 0.0f);
}

TEST(ImageCuda, Sobel) {
    using namespace open3d;
    CheckGradient<Vector1s>(kDepthPath);
    CheckGradient<Vector3b>(kColorPath);
    CheckGradient<Vector1b>(kGrayPath);
}

TEST(ImageCuda, Gaussian) {
    using namespace open3d;
    CheckGaussian<Vector1s>(kDepthPath);
    CheckGaussian<Vector3b>(kColorPath);
    CheckGaussian<Vector1b>(kGrayPath);
}

TEST(ImageCuda, Bilateral) {
    using namespace open3d;
    CheckBilateral<Vector1s>(kDepthPath);
    CheckBilateral<Vector3b>(kColorPath);
    CheckBilateral<Vector1b>(kGrayPath);
}

TEST(ImageCuda, Pyramid) {
    using namespace open3d;
    CheckPyramid<Vector1s, 4>(kDepthPath);
    CheckPyramid<Vector3b, 4>(kColorPath);
    CheckPyramid<Vector1b, 4>(kGrayPath);
}

TEST(ImageCuda, Shift) {
    using namespace open3d;
    CheckShift<Vector1s>(kDepthPath);
    CheckShift<Vector3b>(kColorPath);
    CheckShift<Vector1b>(kGrayPath);
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}