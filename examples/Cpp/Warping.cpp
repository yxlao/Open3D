// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// The MIT License (MIT)
//
// Copyright (c) 2018 www.open3d.org
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
// FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
// IN THE SOFTWARE.
// ----------------------------------------------------------------------------

#include <cstdio>
#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>

#include "Open3D/Open3D.h"

using namespace open3d;

std::vector<color_map::ImageWarpingField> InitWarpingFields(
        const std::vector<std::shared_ptr<geometry::Image>>& images,
        size_t num_vertical_anchors) {
    std::vector<color_map::ImageWarpingField> warping_fields;
    for (auto i = 0; i < images.size(); i++) {
        int width = images[i]->width_;
        int height = images[i]->height_;
        warping_fields.push_back(color_map::ImageWarpingField(
                width, height, num_vertical_anchors));
    }
    return std::move(warping_fields);
}

std::shared_ptr<geometry::Image> ComputeWarpedImage(
        const geometry::Image& im,
        const color_map::ImageWarpingField& warp_field) {
    int width = im.width_;
    int height = im.height_;
    int num_of_channels = im.num_of_channels_;
    int bytes_per_channel = im.bytes_per_channel_;

    auto im_warped = std::make_shared<geometry::Image>();
    im_warped->Prepare(width, height, num_of_channels, bytes_per_channel);

    for (size_t u = 0; u < width; u++) {
        for (size_t v = 0; v < height; v++) {
            Eigen::Vector2d u_v_warp = warp_field.GetImageWarpingField(u, v);
            float u_warp = u_v_warp(0);
            float v_warp = u_v_warp(1);

            bool valid;
            float pixel_val;
            std::tie(valid, pixel_val) = im.FloatValueAt(u_warp, v_warp);
            if (valid) {
                *(im_warped->PointerAt<float>(u, v)) = pixel_val;
            } else {
                *(im_warped->PointerAt<float>(u, v)) = 0;
            }
        }
    }
    return im_warped;
}

std::shared_ptr<geometry::Image> ComputeAverageImage(
        const std::vector<std::shared_ptr<geometry::Image>>& im_grays) {
    if (im_grays.size() == 0) {
        return std::make_shared<geometry::Image>();
    }
    int width = im_grays[0]->width_;
    int height = im_grays[0]->height_;
    int num_of_channels = im_grays[0]->num_of_channels_;
    int bytes_per_channel = im_grays[0]->bytes_per_channel_;
    size_t num_images = im_grays.size();

    auto im_avg = std::make_shared<geometry::Image>();
    im_avg->Prepare(width, height, num_of_channels, 4);
    for (int u = 0; u < width; ++u) {
        for (int v = 0; v < height; ++v) {
            *(im_avg->PointerAt<float>(u, v)) = 0;
            for (const auto& im_gray : im_grays) {
                *(im_avg->PointerAt<float>(u, v)) +=
                        *(im_gray->PointerAt<float>(u, v));
            }
            *(im_avg->PointerAt<float>(u, v)) /= num_images;
        }
    }
    return im_avg;
}

void OptimizeWarpingFields(
        const std::vector<std::shared_ptr<geometry::Image>>& im_grays,
        std::vector<color_map::ImageWarpingField>& warping_fields,
        size_t num_iter) {
    if (im_grays.size() == 0) {
        return;
    }
    int width = im_grays[0]->width_;
    int height = im_grays[0]->height_;
    int num_of_channels = im_grays[0]->num_of_channels_;
    int bytes_per_channel = im_grays[0]->bytes_per_channel_;
    size_t num_images = im_grays.size();

    for (color_map::ImageWarpingField& wf : warping_fields) {
        int num_anchors = wf.GetNumberOfAnchors();
        for (size_t i = 0; i < num_anchors; ++i) {
            wf.flow_(i) = wf.flow_(i) + 100;
        }
    }
}

std::shared_ptr<geometry::Image> ComputeWarpedAverage(
        const std::vector<std::shared_ptr<geometry::Image>>& im_grays,
        const std::vector<color_map::ImageWarpingField>& warping_fields) {
    if (im_grays.size() != warping_fields.size()) {
        throw std::runtime_error("im_grays.size() != warping_fields.size()");
    }

    std::vector<std::shared_ptr<geometry::Image>> im_warps;
    for (size_t i = 0; i < im_grays.size(); i++) {
        im_warps.push_back(ComputeWarpedImage(*im_grays[i], warping_fields[i]));
    }

    return ComputeAverageImage(im_warps);
}

std::vector<std::shared_ptr<geometry::Image>> ReadDataset(
        const std::string& root_dir,
        const std::string& pattern,
        int num_images) {
    std::vector<std::shared_ptr<geometry::Image>> im_grays;
    for (int i = 0; i < num_images; i++) {
        // Get path
        char im_path_buf[1000];
        int status = sprintf(im_path_buf, ("%s/" + pattern).c_str(),
                             root_dir.c_str(), i);
        if (status < 0) {
            throw std::runtime_error("Path formatting error.");
        }
        std::string im_path(im_path_buf);
        std::cout << "Reading: " << im_path << std::endl;

        // Read image and convert to grayscale
        auto im_gray = std::make_shared<geometry::Image>();
        io::ReadImage(im_path, *im_gray);
        im_grays.push_back(im_gray->CreateFloatImage());
    }
    return im_grays;
}

int main(int argc, char** args) {
    // Data path
    utility::SetVerbosityLevel(utility::VerbosityLevel::VerboseAlways);
    std::string im_dir = "/home/ylao/data/inverse-projection";
    std::cout << "im_dir: " << im_dir << std::endl;

    // Read images
    std::vector<std::shared_ptr<geometry::Image>> im_grays =
            ReadDataset(im_dir, "delta-color-%d.png", 33);

    int width = im_grays[0]->width_;
    int height = im_grays[0]->height_;
    int num_of_channels = im_grays[0]->num_of_channels_;
    int bytes_per_channel = im_grays[0]->bytes_per_channel_;
    std::cout << "width: " << width << "\n";
    std::cout << "height: " << height << "\n";
    std::cout << "num_of_channels: " << num_of_channels << "\n";
    std::cout << "bytes_per_channel: " << bytes_per_channel << "\n";

    // Compute average image
    auto im_avg = ComputeAverageImage(im_grays);
    std::string im_avg_path = im_dir + "/avg.png";
    io::WriteImage(im_avg_path, *im_avg->CreateImageFromFloatImage<uint8_t>());

    // Init warping fields
    size_t num_vertical_anchors = 16;
    std::vector<color_map::ImageWarpingField> warping_fields =
            InitWarpingFields(im_grays, num_vertical_anchors);

    // Optimize warping fields
    size_t num_iter = 100;
    OptimizeWarpingFields(im_grays, warping_fields, num_iter);

    // Ouput optimized image
    auto im_warp_avg = ComputeWarpedAverage(im_grays, warping_fields);
    std::string im_warp_avg_path = im_dir + "/avg_warp.png";
    io::WriteImage(im_warp_avg_path,
                   *im_warp_avg->CreateImageFromFloatImage<uint8_t>());

    return 0;
}
