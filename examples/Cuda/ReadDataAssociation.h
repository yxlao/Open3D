//
// Created by wei on 11/28/18.
//

#pragma once

#include <string>
#include <sstream>
#include <fstream>
#include <vector>

#include <Core/Core.h>

inline
std::vector<std::pair<std::string, std::string>>
ReadDataAssociation(const std::string &association_path) {
    std::vector<std::pair<std::string, std::string>> filenames;

    std::ifstream fin(association_path);
    if (!fin.is_open()) {
        open3d::PrintError("Cannot open file %s, abort.\n",
                           association_path.c_str());
        return filenames;
    }

    std::string depth_path;
    std::string image_path;
    while (fin >> depth_path >> image_path) {
        filenames.emplace_back(std::make_pair(depth_path, image_path));
    }

    return filenames;
}