//
// Created by wei on 9/27/18.
//

#include <Open3D/Open3D.h>
#include <Cuda/Geometry/ImageCuda.h>
#include <opencv2/opencv.hpp>
#include <gtest/gtest.h>

using namespace open3d;
using namespace open3d::utility;
using namespace open3d::cuda;

template<typename T, size_t N>
void CheckUploadAndDownloadConsistency(const std::string &path) {
    cv::Mat image = cv::imread(path, cv::IMREAD_UNCHANGED);
    cv::imshow("raw", image);
    cv::waitKey(10);

    Timer timer;
    ImageCuda<T, N> image_cuda, image_cuda_copy;

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

template<typename T, size_t N>
void CheckDownsampling(const std::string &path) {
    using namespace open3d;
    cv::Mat image = cv::imread(path, cv::IMREAD_UNCHANGED);

    Timer timer;
    ImageCuda<T, N> image_cuda, image_cuda_low;

    timer.Start();
    image_cuda.Upload(image);
    timer.Stop();
    PrintInfo("Upload finished in %.3f milliseconds...\n", timer.GetDuration());

    timer.Start();
    image_cuda_low = image_cuda.Downsample(BoxFilter);
    timer.Stop();
    PrintInfo("Downsample finished in %.3f milliseconds...\n", timer.GetDuration());

    timer.Start();
    cv::Mat downloaded = image_cuda_low.DownloadMat();
    timer.Stop();
    PrintInfo("Download finished in %.3f milliseconds...\n", timer.GetDuration());

    cv::imshow("downsampled", downloaded);
    cv::waitKey(10);
    cv::destroyAllWindows();
}

template<typename T, size_t N>
void CheckGaussian(const std::string &path) {
    using namespace open3d;
    cv::Mat image = cv::imread(path, cv::IMREAD_UNCHANGED);

    Timer timer;
    ImageCuda<T, N> image_cuda;

    timer.Start();
    image_cuda.Upload(image);
    timer.Stop();
    PrintInfo("Upload finished in %.3f milliseconds...\n", timer.GetDuration());

    timer.Start();
    ImageCuda<T, N> image_cuda_blurred = image_cuda.Gaussian(Gaussian3x3);
    timer.Stop();
    PrintInfo("Gaussian3x3 finished in %.3f milliseconds...\n",
              timer.GetDuration());

    cv::Mat downloaded = image_cuda_blurred.DownloadMat();
    cv::imshow("Gaussian3x3", downloaded);
    cv::waitKey(10);

    timer.Start();
    image_cuda_blurred = image_cuda.Gaussian(Gaussian5x5);
    timer.Stop();
    PrintInfo("Gaussian5x5 finished in %.3f milliseconds...\n",
              timer.GetDuration());

    downloaded = image_cuda_blurred.DownloadMat();
    cv::imshow("Gaussian5x5", downloaded);
    cv::waitKey(10);

    timer.Start();
    image_cuda_blurred = image_cuda.Gaussian(Gaussian7x7);
    timer.Stop();
    PrintInfo("Gaussian7x7 finished in %.3f milliseconds...\n",
              timer.GetDuration());

    downloaded = image_cuda_blurred.DownloadMat();
    cv::imshow("Gaussian7x7", downloaded);
    cv::waitKey(10);
    cv::destroyAllWindows();
}

template<typename T, size_t N>
void CheckBilateral(const std::string &path) {
    using namespace open3d;
    cv::Mat image = cv::imread(path, cv::IMREAD_UNCHANGED);

    Timer timer;
    ImageCuda<T, N> image_cuda, filtered_image_cuda;

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
    cv::waitKey(10);
}

template<typename T, size_t N>
void CheckToFloatConversion(const std::string &path, float scale, float offset) {
    using namespace open3d;
    cv::Mat image = cv::imread(path, cv::IMREAD_UNCHANGED);

    Timer timer;
    ImageCuda<T, N> image_cuda;
    ImageCuda<float, N> imagef_cuda;
    timer.Start();
    image_cuda.Upload(image);
    timer.Stop();
    PrintInfo("Upload finished in %.3f milliseconds...\n", timer.GetDuration());

    int iter = 100;
    timer.Start();
    for (int i = 0; i < iter; ++i) {
        imagef_cuda = image_cuda.ConvertToFloat(scale, offset);
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
    cv::waitKey(10);
    cv::destroyAllWindows();
}

template<typename T, size_t N>
void CheckGradient(const std::string &path) {
    using namespace open3d;
    cv::Mat image = cv::imread(path, cv::IMREAD_UNCHANGED);

    Timer timer;
    ImageCuda<T, N> image_cuda;
    timer.Start();
    image_cuda.Upload(image);
    timer.Stop();
    PrintInfo("Upload finished in %.3f milliseconds...\n", timer
    .GetDuration());

    timer.Start();
    auto gradients = image_cuda.Sobel();
    timer.Stop();
    PrintInfo("Gradient finished in %.3f milliseconds...\n",
              timer.GetDuration());
    auto dx = std::get<0>(gradients);
    auto dy = std::get<1>(gradients);

    cv::Mat downloaded_dx = dx.DownloadMat();
    cv::Mat downloaded_dy = dy.DownloadMat();
    cv::imshow("dx", downloaded_dx / 255.0f);
    cv::imshow("dy", downloaded_dy / 255.0f);
    cv::waitKey(10);
    cv::destroyAllWindows();
}

template<typename T, size_t N>
void CheckShift(const std::string &path) {
    using namespace open3d;
    cv::Mat image = cv::imread(path, cv::IMREAD_UNCHANGED);

    Timer timer;
    ImageCuda<T, N> image_cuda;
    timer.Start();
    image_cuda.Upload(image);
    timer.Stop();
    PrintInfo("Upload finished in %.3f milliseconds...\n", timer
    .GetDuration());

    timer.Start();
    ImageCuda<T, N> shifted_image = image_cuda.Shift(-120.2f, 135.8f);
    timer.Stop();
    PrintInfo("Shifting finished in %.3f milliseconds...\n",
              timer.GetDuration());

    cv::Mat downloaded = shifted_image.DownloadMat();
    cv::imshow("shifted", downloaded);
    cv::waitKey(10);
    cv::destroyAllWindows();
}

void CheckRGBToIntensity(const std::string &path) {
    using namespace open3d;
    cv::Mat image = cv::imread(path, cv::IMREAD_UNCHANGED);
    cv::cvtColor(image, image, cv::COLOR_BGR2RGB);

    cv::Mat intensity;
    cv::cvtColor(image, intensity, cv::COLOR_RGB2GRAY);

    ImageCuda<uchar, 3> image_cuda;
    image_cuda.Upload(image);
    ImageCuda<float, 1> intensity_cuda = image_cuda.ConvertRGBToIntensity();

    cv::imshow("intensity", intensity_cuda.DownloadMat());
    cv::imshow("intensity cv", intensity);
    cv::waitKey(-1);
}

const std::string kDepthPath = "../../../examples/TestData/RGBD/other_formats/TUM_depth.png";
const std::string kColorPath = "../../../examples/TestData/RGBD/other_formats/TUM_color.png";
const std::string kGrayPath  = "../../../examples/TestData/lena_gray.jpg";

TEST(ImageCuda, ConvertRGBToIntensity) {
    using namespace open3d;
    CheckRGBToIntensity(kColorPath);
}

TEST(ImageCuda, UploadAndDownload) {
    using namespace open3d;
    CheckUploadAndDownloadConsistency<ushort, 1>(kDepthPath);
    CheckUploadAndDownloadConsistency<uchar, 3>(kColorPath);
    CheckUploadAndDownloadConsistency<uchar, 1>(kGrayPath);
}

TEST(ImageCuda, Downsampling) {
    using namespace open3d;
    CheckDownsampling<ushort, 1>(kDepthPath);
    CheckDownsampling<uchar, 3>(kColorPath);
    CheckDownsampling<uchar, 1>(kGrayPath);
}

TEST(ImageCuda, ToFloatConversion) {
    using namespace open3d;
    CheckToFloatConversion<ushort, 1>(kDepthPath, 1.0f / 5000.0f, 0.0f);
    CheckToFloatConversion<uchar, 3>(kColorPath, 1.0f / 255.0f, 0.0f);
    CheckToFloatConversion<uchar, 1>(kGrayPath, 1.0f / 255.0f, 0.0f);
}

TEST(ImageCuda, Sobel) {
    using namespace open3d;
    CheckGradient<ushort, 1>(kDepthPath);
    CheckGradient<uchar, 3>(kColorPath);
    CheckGradient<uchar, 1>(kGrayPath);
}

TEST(ImageCuda, Gaussian) {
    using namespace open3d;
    CheckGaussian<ushort, 1>(kDepthPath);
    CheckGaussian<uchar, 3>(kColorPath);
    CheckGaussian<uchar, 1>(kGrayPath);
}

TEST(ImageCuda, Bilateral) {
    using namespace open3d;
    CheckBilateral<ushort, 1>(kDepthPath);
    CheckBilateral<uchar, 3>(kColorPath);
    CheckBilateral<uchar, 1>(kGrayPath);
}

TEST(ImageCuda, Shift) {
    using namespace open3d;
    CheckShift<ushort, 1>(kDepthPath);
    CheckShift<uchar, 3>(kColorPath);
    CheckShift<uchar, 1>(kGrayPath);
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}