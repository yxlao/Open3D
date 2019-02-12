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
    cv::Mat color;
};

std::tuple<bool, Eigen::Matrix4d> PoseEstimationPnP(
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
                       false, 1000, 1.0, 0.999, inliers);

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

std::tuple<bool, Eigen::Matrix4d> PointToPointICPRansac(
    std::vector<Eigen::Vector3d> &pts3d_source,
    std::vector<Eigen::Vector3d> &pts3d_target,
    std::vector<int> &indices) {

    const int kMaxIter = 1000;
    const int kNumSamples = 5;
    const int kNumPoints = pts3d_source.size();
    const double kMaxDistance = 0.05;

    bool success = false;
    Eigen::Matrix4d best_trans = Eigen::Matrix4d::Identity();
    int max_inlier = 5;

    for (int i = 0; i < kMaxIter; ++i) {
        std::random_shuffle(indices.begin(), indices.end());

        /** Sample **/
        Eigen::MatrixXd pts3d_source_sample(3, kNumSamples);
        Eigen::MatrixXd pts3d_target_sample(3, kNumSamples);
        for (int j = 0; j < kNumSamples; ++j) {
            pts3d_source_sample.col(j) = pts3d_source[indices[j]];
            pts3d_target_sample.col(j) = pts3d_target[indices[j]];
        }

        /** Compute model **/
        Eigen::Matrix4d trans = Eigen::umeyama(
            pts3d_source_sample, pts3d_target_sample, false);

        /** Evaluate **/
        int inlier = 0;
        for (int j = 0; j < kNumPoints; ++j) {
            Eigen::Vector3d pt3d_source_on_target =
                (trans * pts3d_source[j].homogeneous()).hnormalized();
            if ((pt3d_source_on_target - pts3d_target[j]).norm()
                < kMaxDistance) {
                ++inlier;
            }
        }

        if (inlier > max_inlier) {
            success = true;
            best_trans = trans;
            max_inlier = inlier;
        }
    }

//    std::cout << best_trans << "\n";
    return std::make_tuple(success, best_trans);
}

std::tuple<bool, Eigen::Matrix4d> PoseEstimation(
    KeyframeInfo &source_info, KeyframeInfo &target_info,
    PinholeCameraIntrinsic &intrinsic) {

    const int kNumSamples = 5;

    /** No descriptor **/
    if (source_info.descriptor.empty() || target_info.descriptor.empty()) {
        return std::make_tuple(false, Eigen::Matrix4d::Identity());
    }

    cv::Ptr<cv::BFMatcher> matcher =
        cv::BFMatcher::create(cv::NORM_HAMMING, true);
    std::vector<cv::DMatch> matches;
    matcher->match(source_info.descriptor, target_info.descriptor, matches);

    /** Unable to perform ICP **/
    if (matches.size() < kNumSamples) {
        return std::make_tuple(false, Eigen::Matrix4d::Identity());
    }

    std::vector<cv::Point2f> pts2d_source;
    std::vector<cv::Point2f> pts2d_target;
//    std::cout << matches.size() << std::endl;
    for (auto &match : matches) {
//        std::cout << source_info.keypoints[match.queryIdx].pt
//                  << " - "
//                  << target_info.keypoints[match.trainIdx].pt << std::endl;
        pts2d_source.push_back(source_info.keypoints[match.queryIdx].pt);
        pts2d_target.push_back(target_info.keypoints[match.trainIdx].pt);
    }

    cv::Matx33d K = cv::Matx33d::zeros();
    K(2, 2) = 1;
    std::tie(K(0, 0), K(1, 1)) = intrinsic.GetFocalLength();
    std::tie(K(0, 2), K(1, 2)) = intrinsic.GetPrincipalPoint();

    cv::Mat mask;
    cv::findEssentialMat(pts2d_source, pts2d_target, K,
                         cv::RANSAC, 0.999, 1.0, mask);

//    cv::Mat out;
//    cv::drawMatches(source_info.color, source_info.keypoints,
//                    target_info.color, target_info.keypoints,
//                    matches, out,
//                    cv::Scalar::all(-1), cv::Scalar::all(-1), mask);
//    cv::imshow("out", out);
//    cv::waitKey(-1);


    std::vector<Eigen::Vector3d> pts3d_source;
    std::vector<Eigen::Vector3d> pts3d_target;
    std::vector<int> indices; // for shuffle
    Eigen::Matrix3d inv_K = intrinsic.intrinsic_matrix_.inverse();

    int cnt = 0;
    for (int i = 0; i < matches.size(); ++i) {
        if (mask.at<char>(i)) {
            Eigen::Vector2d pt2d_source = Eigen::Vector2d(
                pts2d_source[i].x, pts2d_source[i].y);
            Eigen::Vector2d pt2d_target = Eigen::Vector2d(
                pts2d_target[i].x, pts2d_target[i].y);

            float depth_source = source_info.depth.at<float>(
                pt2d_source(1), pt2d_source(0));
            float depth_target = target_info.depth.at<float>(
                pt2d_target(1), pt2d_target(0));

            if (depth_source > 0) {
                pts3d_source.emplace_back(depth_source * (inv_K *
                    Eigen::Vector3d(pt2d_source(0), pt2d_source(1), 1)));
            } else {
                pts3d_source.emplace_back(Eigen::Vector3d::Zero());
            }

            if (depth_target > 0) {
                pts3d_target.emplace_back(depth_target * (inv_K *
                    Eigen::Vector3d(pt2d_target(0), pt2d_target(1), 1)));
            } else {
                pts3d_target.emplace_back(Eigen::Vector3d::Zero());
            }

//            std::cout << cnt << ": ";
//            std::cout << pt2d_source.transpose() << " -> ";
//            std::cout << pts3d_source[cnt].transpose() << " vs ";
//
//            std::cout << pt2d_target.transpose() << " -> ";
//            std::cout << pts3d_target[cnt].transpose() << "\n";

            indices.push_back(cnt++);
        }
    }

    /** Unable to perform ICP **/
    if (indices.size() < kNumSamples) {
        return std::make_tuple(false, Eigen::Matrix4d::Identity());
    }

    return PointToPointICPRansac(pts3d_source, pts3d_target, indices);
}
}