//
// Created by wei on 2/4/19.
//

#pragma once

#include <string>
#include <vector>
#include <iomanip>
#include <sstream>

const int kFramesPerFragment = 100;
const float kCubicSize = 3.0f;
const float kTSDFTruncation = 0.04f;
const float kVoxelSize = 0.05f;

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