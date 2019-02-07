//
// Created by wei on 2/4/19.
//

#pragma once

#include <string>
#include <vector>
#include <iomanip>
#include <sstream>

//const std::string kBasePath = "/home/wei/Work/data/stanford/lounge";
//const std::string kBasePath = "/home/wei/Work/data/stanford/copyroom";
const std::string kBasePath
    ="/media/wei/Data/data/indoor_lidar_rgbd/apartment";
const int kNumFragments = 30;

const int kFramesPerFragment = 100;

const float kCubicSize = 3.0f;
const float kTSDFTruncation = 0.04f;

const float kVoxelSize = 0.05f;
const float kMaxDepthDiff = 0.07f;
const float kDepthMin = 0.05f;
const float kDepthMax = 4.0f;
const float kDepthFactor = 1000.0f;

const float kPreferenceLoopClosureOdometry = 0.25f;
const float kPreferenceLoopClosureRegistration = 5.0f;


struct Match {
    bool success;
    int s;
    int t;
    Eigen::Matrix4d trans_source_to_target;
    Eigen::Matrix6d information;
};


std::string GetFragmentPoseGraphName(
    int fragment_id,
    const std::string &base_path,
    const std::string &subfix = "") {
    std::stringstream ss;
    ss << base_path << "/fragments_cuda/fragment_" << subfix;
    ss << std::setw(3) << std::setfill('0') << fragment_id;
    ss << ".json";

    return ss.str();
}

std::string GetScenePoseGraphName(
    const std::string &base_path,
    const std::string &subfix = "" /* "_optimized" */) {
    std::stringstream ss;
    ss << base_path << "/scene_cuda/global_registration" << subfix;
    ss << ".json";

    return ss.str();
}

std::string GetFragmentPlyName(
    int fragment_id,
    const std::string &base_path) {
    std::stringstream ss;
    ss << base_path << "/fragments_cuda/fragment_";
    ss << std::setw(3) << std::setfill('0') << fragment_id;
    ss << ".ply";

    return ss.str();
}

std::vector<std::string> GetFragmentPlyNames(
    const std::string &base_path,
    const int num_files) {

    std::vector<std::string> filenames;
    for (int i = 0; i < num_files; ++i) {
        filenames.emplace_back(GetFragmentPlyName(i, base_path));
    }
    return filenames;
}

std::string GetScenePlyName(
    const std::string &base_path) {
    std::stringstream ss;
    ss << base_path << "/scene_cuda/integrated.ply";

    return ss.str();
}