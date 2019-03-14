//
// Created by wei on 2/8/19.
//

#include <Open3D/Open3D.h>

#include <opencv2/opencv.hpp>
#include <Cuda/Geometry/RGBDImageCuda.h>
#include <Cuda/Geometry/PointCloudCuda.h>
#include "examples/Cuda/DatasetConfig.h"

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

cv::Mat drawMatches(
    std::vector<cv::Point3f> &pts3d_source,
    std::vector<cv::Point2f> &pts2d_source,
    std::vector<cv::Point2f> &pts2d_target,
    std::vector<int> &inliers,
    const Eigen::Matrix3d &intrinsics,
    const Eigen::Matrix4d &extrinsics,
    cv::Mat &left, cv::Mat &right) {

    cv::Mat out(left.rows, left.cols + right.cols, CV_8UC1);
    cv::Mat tmp = out(cv::Rect(0, 0, left.cols, left.rows));
    left.copyTo(tmp);
    tmp = out(cv::Rect(left.cols, 0, right.cols, right.rows));
    right.copyTo(tmp);
    cv::cvtColor(out, out, CV_GRAY2BGR);

    for (int j = 0; j < inliers.size(); ++j) {
        int i = inliers[j];
        cv::Point3f pt = pts3d_source[i];
        Eigen::Vector4d pt3d_source(pt.x, pt.y, pt.z, 1);
        Eigen::Vector2d pt2d_source_on_target = (intrinsics *
            (extrinsics * pt3d_source).hnormalized()).hnormalized();

        auto pt_s = pts2d_source[i];
        auto pt_s_o_t = cv::Point2f(pt2d_source_on_target(0) + left.cols,
                                    pt2d_source_on_target(1));
        auto pt_t = cv::Point2f(pts2d_target[i].x + left.cols,
                                pts2d_target[i].y);

        cv::circle(out, pt_s, 3, cv::Vec3b(255, 0, 0));
        cv::circle(out, pt_s_o_t, 3, cv::Vec3b(255, 0, 0));
        cv::line(out, pt_s, pt_s_o_t, cv::Vec3b(255, 0, 0));

        cv::circle(out, pt_t, 3, cv::Vec3b(0, 0, 255));
        cv::line(out, pt_s_o_t, pt_t, cv::Vec3b(0, 0, 255));
    }

    return out;
}

std::tuple<bool, Eigen::Matrix4d> PoseEstimationPnP(
    KeyframeInfo &source_info, KeyframeInfo &target_info,
    camera::PinholeCameraIntrinsic &intrinsic) {

    /** No descriptor **/
    if (source_info.descriptor.empty() || target_info.descriptor.empty()) {
        return std::make_tuple(false, Eigen::Matrix4d::Identity());
    }

    cv::Ptr<cv::BFMatcher> matcher =
        cv::BFMatcher::create(cv::NORM_HAMMING, true);
    std::vector<cv::DMatch> matches;
    matcher->match(source_info.descriptor, target_info.descriptor, matches);

    cv::Mat o;
    cv::drawMatches(source_info.color, source_info.keypoints,
                    target_info.color, target_info.keypoints,
                    matches, o);
    cv::imshow("o", o);
    cv::waitKey(-1);

    /** Unable to perform pnp **/
    if (matches.size() < 4) {
        return std::make_tuple(false, Eigen::Matrix4d::Identity());
    }

    std::vector<cv::Point3f> pts3d_source;
    std::vector<cv::Point2f> pts2d_source;
    std::vector<cv::Point2f> pts2d_target;
    std::map<int, int> indices;
    Eigen::Matrix3d inv_intrinsic = intrinsic.intrinsic_matrix_.inverse();

    int cnt = 0;
    for (int i = 0; i < matches.size(); ++i) {
        auto &match = matches[i];
        cv::Point2f uv_source = source_info.keypoints[match.queryIdx].pt;
        cv::Point2f uv_target = target_info.keypoints[match.trainIdx].pt;
        int u = (int) uv_source.x;
        int v = (int) uv_source.y;

        float d = source_info.depth.at<float>(v, u);
        if (!std::isnan(d) && d > 0) {
            Eigen::Vector3d pt(u, v, 1);
            Eigen::Vector3d pt3d = d * (inv_intrinsic * pt);
            std::cout << pt3d.transpose() << std::endl;
            pts3d_source.emplace_back(cv::Point3f(pt3d(0), pt3d(1), pt3d(2)));
            pts2d_source.emplace_back(uv_source);
            pts2d_target.emplace_back(uv_target);

            indices[pts3d_source.size() - 1] = i;
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
                       false, 1000, 3.0, 0.999, inliers);

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

    cv::Mat out = drawMatches(
        pts3d_source, pts2d_source, pts2d_target, inliers,
        intrinsic.intrinsic_matrix_, transform,
        source_info.color, target_info.color);
    cv::imshow("out", out);
    cv::waitKey(-1);

    utility::PrintDebug("Matches %d: PnP: %d, Inliers: %d\n",
               matches.size(), pts3d_source.size(), inliers.size());
    return std::make_tuple(true, transform);
}

cv::Mat drawMatches(
    cv::Mat &left, cv::Mat &right,
    std::vector<Eigen::Vector3d> &pts3d_source,
    std::vector<Eigen::Vector2d> &pts2d_source,
    std::vector<Eigen::Vector3d> &pts3d_target,
    std::vector<Eigen::Vector2d> &pts2d_target,
    const Eigen::Matrix3d &intrinsics,
    const Eigen::Matrix4d &extrinsics,
    std::vector<char> &inliers) {

    cv::Mat out(left.rows, left.cols + right.cols, CV_8UC1);
    cv::Mat tmp = out(cv::Rect(0, 0, left.cols, left.rows));
    left.copyTo(tmp);
    tmp = out(cv::Rect(left.cols, 0, right.cols, right.rows));
    right.copyTo(tmp);
    cv::cvtColor(out, out, CV_GRAY2BGR);

    for (int j = 0; j < inliers.size(); ++j) {
        if (inliers[j]) continue;

        Eigen::Vector3d pt3d_source = pts3d_source[j];
        Eigen::Vector3d pt3d_target = pts3d_target[j];
        auto tmp_s = (intrinsics * pt3d_source).hnormalized();
        auto tmp_t = (intrinsics * pt3d_target).hnormalized();

        Eigen::Vector2d pt2d_source_on_target = (intrinsics *
            (extrinsics * pt3d_source.homogeneous()).hnormalized()).hnormalized();

        auto pt_s = cv::Point2f(pts2d_source[j](0), pts2d_source[j](1));
        auto pt_s_proj = cv::Point2f(tmp_s(0), tmp_s(1));

        auto pt_t = cv::Point2f(pts2d_target[j](0) + left.cols,pts2d_target[j](1));
        auto pt_t_proj = cv::Point2f(tmp_t(0) + left.cols, tmp_t(1));

        auto pt_s_o_t = cv::Point2f(pt2d_source_on_target(0) + left.cols,
                                    pt2d_source_on_target(1));

        cv::circle(out, pt_s_proj, 5, cv::Vec3b(0, 255, 0));
        cv::circle(out, pt_t_proj, 5, cv::Vec3b(0, 255, 0));

        cv::circle(out, pt_s, 3, cv::Vec3b(255, 0, 0));
        cv::circle(out, pt_t, 3, cv::Vec3b(0, 0, 255));

        cv::circle(out, pt_s_o_t, 3, cv::Vec3b(255, 0, 0));

        cv::line(out, pt_s, pt_s_o_t, cv::Vec3b(255, 0, 0));
        cv::line(out, pt_s_o_t, pt_t, cv::Vec3b(0, 0, 255));
    }

    return out;
}

std::tuple<bool, Eigen::Matrix4d, std::vector<char>> PointToPointICPRansac(
    std::vector<Eigen::Vector3d> &pts3d_source,
    std::vector<Eigen::Vector3d> &pts3d_target,
    std::vector<int> &indices) {

    const int kMaxIter = 1000;
    const int kNumSamples = 5;
    const int kNumPoints = pts3d_source.size();
    const double kMaxDistance = 0.05;

    bool success = false;
    Eigen::Matrix4d best_trans = Eigen::Matrix4d::Identity();
    std::vector<char> best_inliers;
    best_inliers.resize(indices.size());

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
        std::vector<char> inliers;
        inliers.resize(best_inliers.size());

        for (int j = 0; j < kNumPoints; ++j) {
            Eigen::Vector3d pt3d_source_on_target =
                (trans * pts3d_source[j].homogeneous()).hnormalized();
            if ((pt3d_source_on_target - pts3d_target[j]).norm()
                < kMaxDistance) {
                ++inlier;
            } else inliers[j] = 0;
        }

        if (inlier > max_inlier) {
            success = true;
            best_trans = trans;
            max_inlier = inlier;
            best_inliers = inliers;
        }
    }

    return std::make_tuple(success, best_trans, best_inliers);
}

std::tuple<bool, Eigen::Matrix4d> PoseEstimation(
    KeyframeInfo &source_info, KeyframeInfo &target_info,
    camera::PinholeCameraIntrinsic &intrinsic, bool debug = false) {

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

    std::vector<cv::Point2f> pts2d_source_kp;
    std::vector<cv::Point2f> pts2d_target_kp;
    for (auto &match : matches) {
        pts2d_source_kp.push_back(source_info.keypoints[match.queryIdx].pt);
        pts2d_target_kp.push_back(target_info.keypoints[match.trainIdx].pt);
    }

    cv::Matx33d K = cv::Matx33d::zeros();
    K(2, 2) = 1;
    std::tie(K(0, 0), K(1, 1)) = intrinsic.GetFocalLength();
    std::tie(K(0, 2), K(1, 2)) = intrinsic.GetPrincipalPoint();
    cv::Mat mask;
    cv::findEssentialMat(pts2d_source_kp, pts2d_target_kp, K,
                         cv::RANSAC, 0.999, 1.0, mask);

    std::vector<Eigen::Vector3d> pts3d_source;
    std::vector<Eigen::Vector2d> pts2d_source;
    std::vector<Eigen::Vector3d> pts3d_target;
    std::vector<Eigen::Vector2d> pts2d_target;

    std::vector<int> indices; // for shuffle
    Eigen::Matrix3d inv_K = intrinsic.intrinsic_matrix_.inverse();

    int cnt = 0;
    for (int i = 0; i < matches.size(); ++i) {
        if (mask.at<char>(i)) {
            Eigen::Vector2d pt2d_source = Eigen::Vector2d(
                pts2d_source_kp[i].x, pts2d_source_kp[i].y);
            Eigen::Vector2d pt2d_target = Eigen::Vector2d(
                pts2d_target_kp[i].x, pts2d_target_kp[i].y);

            float depth_source = source_info.depth.at<float>(
                pt2d_source(1), pt2d_source(0));
            float depth_target = target_info.depth.at<float>(
                pt2d_target(1), pt2d_target(0));

            if (!std::isnan(depth_source) && depth_source > 0
                && !std::isnan(depth_target) && depth_target > 0) {
                pts3d_source.emplace_back(depth_source * (inv_K *
                    Eigen::Vector3d(pt2d_source(0), pt2d_source(1), 1)));
                pts2d_source.emplace_back(pt2d_source);

                pts3d_target.emplace_back(depth_target * (inv_K *
                Eigen::Vector3d(pt2d_target(0), pt2d_target(1), 1)));
                pts2d_target.emplace_back(pt2d_target);

                indices.push_back(cnt);
                ++cnt;
            }
        }
    }

    /** Unable to perform ICP **/
    if (indices.size() < kNumSamples) {
        return std::make_tuple(false, Eigen::Matrix4d::Identity());
    }

    bool success;
    Eigen::Matrix4d transform;
    std::vector<char> inliers;
    std::tie(success, transform, inliers) =
        PointToPointICPRansac(pts3d_source, pts3d_target, indices);

    if (debug) {
        cv::Mat out = drawMatches(source_info.color, target_info.color,
            pts3d_source, pts2d_source, pts3d_target, pts2d_target,
            intrinsic.intrinsic_matrix_, transform, inliers);
        cv::imshow("out", out);
        cv::waitKey(-1);
    }

    return std::make_tuple(success, transform);
}
}