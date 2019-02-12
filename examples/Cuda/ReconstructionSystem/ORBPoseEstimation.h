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

    /** No descriptor **/
    if (source_info.descriptor.empty() || target_info.descriptor.empty()) {
        return std::make_tuple(false, Eigen::Matrix4d::Identity());
    }

    cv::Ptr<cv::BFMatcher> matcher =
        cv::BFMatcher::create(cv::NORM_HAMMING, true);
    std::vector<cv::DMatch> matches;
    matcher->match(source_info.descriptor, target_info.descriptor, matches);

    /** Unable to perform pnp **/
    if (matches.size() < 4) {
        return std::make_tuple(false, Eigen::Matrix4d::Identity());
    }

    std::vector<cv::Point3f> pts3d_source;
    std::vector<cv::Point2f> pts2d_target;
    Eigen::Matrix3d inv_intrinsic = intrinsic.intrinsic_matrix_.inverse();
    for (auto &match : matches) {
        cv::Point2f uv_source = source_info.keypoints[match.queryIdx].pt;
        cv::Point2f uv_target = target_info.keypoints[match.trainIdx].pt;
        int u = (int) uv_source.x;
        int v = (int) uv_source.y;

        float d = source_info.depth.at<float>(v, u);
        if (d > 0) {
            Eigen::Vector3d pt(u, v, 1);
            Eigen::Vector3d pt3d = d * (inv_intrinsic * pt);
            pts3d_source.emplace_back(cv::Point3f(pt3d(0), pt3d(1), pt3d(2)));
            pts2d_target.emplace_back(uv_target);
        }
    }

    /** Unable to perform pnp **/
    if (pts3d_source.size() < 4) {
        return std::make_tuple(false, Eigen::Matrix4d::Identity());
    }

    cv::Mat rvec, tvec;
    cv::Matx33d K = cv::Matx33d::zeros();
    K(2, 2) = 1;
    std::tie(K(0, 0), K(1, 1)) = intrinsic.GetFocalLength();
    std::tie(K(0, 2), K(1, 2)) = intrinsic.GetPrincipalPoint();

    std::vector<int> inliers;
    cv::solvePnPRansac(pts3d_source, pts2d_target,
                       K, cv::noArray(), rvec, tvec,
                       false, 100, 3.0, 0.999, inliers);

    /** pnp not successful **/
    if (inliers.size() < 4) {
        return std::make_tuple(false, Eigen::Matrix4d::Identity());
    }

    cv::Mat R;
    cv::Rodrigues(rvec, R);
    Eigen::Matrix4d transform = Eigen::Matrix4d::Identity();
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            transform(i, j) = R.at<double>(i, j);
        }
        transform(i, 3) = tvec.at<double>(i);
    }

    PrintDebug("Matches %d: PnP: %d, Inliers: %d\n",
               matches.size(), pts3d_source.size(), inliers.size());
    return std::make_tuple(true, transform);
}
}