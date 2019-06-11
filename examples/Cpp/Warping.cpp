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

#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>

#include "Open3D/Open3D.h"

using namespace open3d;

int main(int argc, char** args) {
    // Data path
    utility::SetVerbosityLevel(utility::VerbosityLevel::VerboseAlways);
    if (argc < 2) {
        PrintOpen3DVersion();
        utility::PrintInfo("Usage: ./Warping [im_path]\n");
        return 1;
    }
    std::string im_dir(args[1]);
    std::cout << "im_dir: " << im_dir << std::endl;

    // Read images
    size_t num_images = 6;
    std::vector<std::shared_ptr<geometry::Image>> im_rgbs;
    for (size_t im_idx = 0; im_idx < num_images; ++im_idx) {
        std::stringstream im_path;
        im_path << im_dir << "/" << std::setw(2) << std::setfill('0') << im_idx
                << ".jpg";
        std::cout << "Reading: " << im_path.str() << std::endl;
        auto im_rgb = std::make_shared<geometry::Image>();
        io::ReadImage(im_path.str(), *im_rgb);
        im_rgbs.push_back(im_rgb->CreateFloatImage());
    }
    std::cout << "width: " << im_rgbs[0]->width_ << "\n";
    std::cout << "height: " << im_rgbs[0]->height_ << "\n";
    std::cout << "num_of_channels: " << im_rgbs[0]->num_of_channels_ << "\n";
    std::cout << "bytes_per_channel: " << im_rgbs[0]->bytes_per_channel_
              << "\n";

    return 0;
}
