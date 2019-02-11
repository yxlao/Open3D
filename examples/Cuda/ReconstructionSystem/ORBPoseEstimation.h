//
// Created by wei on 2/8/19.
//

#include <Core/Core.h>
#include <IO/IO.h>
#include <Visualization/Visualization.h>
#include <opencv2/opencv.hpp>
#include <Cuda/Geometry/RGBDImageCuda.h>
#include <Cuda/Geometry/PointCloudCuda.h>
#include "DatasetConfig.h"

using namespace open3d;

namespace ORBPoseEstimation {

class KeyframeInfo {
public:
    int idx;
    std::vector<cv::KeyPoint> keypoints;
    cv::Mat descriptor;
    cv::Mat depth;
};

std::tuple<bool, Eigen::Matrix4d> PoseEstimation(
    KeyframeInfo &source_info, KeyframeInfo &target_info,
    PinholeCameraIntrinsic &intrinsic) {

    const int kValidPtsThreshold = 20;

    cv::Ptr<cv::DescriptorMatcher> matcher = cv::DescriptorMatcher::create(
        "BruteForce-Hamming");

    std::vector<std::vector<cv::DMatch> > matches;
    std::vector<cv::DMatch> best_matches;
    std::vector<cv::Point2f> matched1, matched2;
    if (source_info.descriptor.empty() || target_info.descriptor.empty()) {
        return std::make_tuple(false, Eigen::Matrix4d::Identity());
    }

    matcher->knnMatch(source_info.descriptor, target_info.descriptor,
        matches, 2);

    for (unsigned i = 0; i < matches.size(); i++) {
        if (matches[i][0].distance < 0.8f * matches[i][1].distance) {
            best_matches.push_back(matches[i][0]);
            matched1.push_back(source_info.keypoints[matches[i][0].queryIdx].pt);
            matched2.push_back(target_info.keypoints[matches[i][0].trainIdx].pt);
        }
    }

    if (best_matches.size() < kValidPtsThreshold) {
        return std::make_tuple(false, Eigen::Matrix4d::Identity());
    }

    std::vector<cv::Point3f> pts3d_source;
    std::vector<cv::Point2f> pts2d_target;
    Eigen::Matrix3d inv_intrinsic = intrinsic.intrinsic_matrix_.inverse();

    for (int i = 0; i < matched1.size(); ++i) {
        cv::Point2f uv_source = matched1[i];
        int u = (int) uv_source.x;
        int v = (int) uv_source.y;

        float d = source_info.depth.at<float>(u, v);
        if (d > 0) {
            Eigen::Vector3d pt;
            pt << u, v, 1;
            Eigen::Vector3d pt3d = d * (inv_intrinsic * pt);
            pts3d_source.emplace_back(cv::Point3f(pt3d(0), pt3d(1), pt3d(2)));
            pts2d_target.push_back(matched2[i]);
        }
    }

    if (pts3d_source.size() < kValidPtsThreshold) {
        return std::make_tuple(false, Eigen::Matrix4d::Identity());
    }

    cv::Mat rvec, tvec, mat_intrinsic = cv::Mat(3, 3, CV_32FC1);
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            mat_intrinsic.at<float>(i, j)
                = intrinsic.intrinsic_matrix_(i, j);
        }
    }

    std::vector<int> inliers;
    cv::solvePnPRansac(pts3d_source, pts2d_target,
                       mat_intrinsic, cv::noArray(), rvec, tvec,
                       false, 100, 3.0, 0.99, inliers);
    if (inliers.size() < kValidPtsThreshold) {
        return std::make_tuple(false, Eigen::Matrix4d::Identity());
    }

    cv::Mat R;
    cv::Rodrigues(rvec, R);

    Eigen::Matrix4d transform = Eigen::Matrix4d::Identity();
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            transform(i, j) = R.at<double>(i, j);
        }
    }
    for (int i = 0; i < 3; ++i) {
        transform(i, 3) = tvec.at<double>(i);
    }

    PrintDebug("Matches %d: pNp: %d, Inliers: %d\n",
        best_matches.size(), pts3d_source.size(), inliers.size());
    return std::make_tuple(true, transform);
}
}

//int main(int argc, char **argv) {
//    DatasetConfig config;
//
//    std::cout << "Reading configs" << std::endl;
//    std::string config_path = argc > 1 ? argv[1] :
//                              "/home/wei/Work/projects/dense_mapping/Open3D/examples/Cuda"
//                              "/ReconstructionSystem/config/loft.json";
//
//    bool is_success = ReadIJsonConvertible(config_path, config);
//    if (!is_success) return 1;
//
//    std::cout << "Preparing images" << std::endl;
//    cuda::RGBDImageCuda rgbd_source((float) config.min_depth_,
//                                    (float) config.max_depth_,
//                                    (float) config.depth_factor_);
//    cuda::RGBDImageCuda rgbd_target((float) config.min_depth_,
//                                    (float) config.max_depth_,
//                                    (float) config.depth_factor_);
//
//    std::cout << "Reading and Uploading images" << std::endl;
//    Image depth, color;
//    ReadImage(config.depth_files_[0], depth);
//    ReadImage(config.color_files_[0], color);
//    rgbd_source.Upload(depth, color);
//
//    ReadImage(config.depth_files_[20], depth);
//    ReadImage(config.color_files_[20], color);
//    rgbd_target.Upload(depth, color);
//
//    PinholeCameraIntrinsic intrinsic = config.intrinsic_;
//
//    Eigen::Matrix4d source_to_target = ORBPoseEstimation::PoseEstimation
//        (rgbd_source, rgbd_target, intrinsic);
//
//    std::shared_ptr<cuda::PointCloudCuda>
//        pcl_source = std::make_shared<cuda::PointCloudCuda>(
//        cuda::VertexWithColor, 300000),
//        pcl_target = std::make_shared<cuda::PointCloudCuda>(
//        cuda::VertexWithColor, 300000);
//
//    cuda::PinholeCameraIntrinsicCuda intrinsic_cuda(intrinsic);
//    pcl_source->Build(rgbd_source, intrinsic_cuda);
//    pcl_target->Build(rgbd_target, intrinsic_cuda);
//    pcl_source->Transform(source_to_target);
//
//    DrawGeometries({pcl_source, pcl_target});
//}