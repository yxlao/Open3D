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

namespace Eigen {
typedef Eigen::Matrix<double, 8, 8> Matrix8d;
typedef Eigen::Matrix<double, 8, 1> Vector8d;
typedef Eigen::Matrix<int, 8, 1> Vector8i;
}  // namespace Eigen

class WarpFieldOptimizerOption {
public:
    WarpFieldOptimizerOption(
            // Attention: when you update the defaults, update the docstrings in
            // Python/color_map/color_map.cpp
            int num_iters = 100,
            int num_vertical_anchors = 10,
            double anchor_weight = 0.316)
        : num_iters_(num_iters),
          num_vertical_anchors_(num_vertical_anchors),
          anchor_weight_(anchor_weight) {}
    ~WarpFieldOptimizerOption() {}

public:
    int num_iters_;
    int num_vertical_anchors_;
    double anchor_weight_;
};

class WarpFieldOptimizer {
public:
    WarpFieldOptimizer(
            const std::vector<std::shared_ptr<geometry::Image>>& im_grays,
            const std::vector<std::shared_ptr<geometry::Image>>& im_masks,
            const WarpFieldOptimizerOption& option)
        : im_grays_(im_grays), im_masks_(im_masks), option_(option) {
        // TODO: ok to throw exception here?
        if (im_grays.size() == 0) {
            throw std::runtime_error("Empty inputs");
        }

        // TODO: check that all images are of the same size
        width_ = im_grays[0]->width_;
        height_ = im_grays[0]->height_;
        num_of_channels_ = im_grays[0]->num_of_channels_;
        num_images_ = im_grays.size();

        // Init warping fields
        for (auto i = 0; i < num_images_; i++) {
            warp_fields_.push_back(color_map::ImageWarpingField(
                    width_, height_, option_.num_vertical_anchors_));
        }
        warp_fields_identity_ = color_map::ImageWarpingField(
                width_, height_, option_.num_vertical_anchors_);
        anchor_w_ = warp_fields_[0].anchor_w_;
        anchor_h_ = warp_fields_[0].anchor_h_;
        anchor_step_ = warp_fields_[0].anchor_step_;

        // Init gradient images
        im_dxs_.clear();
        im_dys_.clear();
        for (const auto& im_gray : im_grays) {
            im_dxs_.push_back(
                    im_gray->Filter(geometry::Image::FilterType::Sobel3Dx));
            im_dys_.push_back(
                    im_gray->Filter(geometry::Image::FilterType::Sobel3Dy));
        }
    }
    ~WarpFieldOptimizer() {}

    // Run optimization of warp_fields_
    void Optimize() {
        // Initialize proxy image with avg
        std::shared_ptr<geometry::Image> im_proxy = ComputeWarpAverageImage();

        for (size_t iter = 0; iter < option_.num_iters_; iter++) {
            double residual_sum = 0.0;
            double residual_reg_sum = 0.0;

#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
            for (size_t im_idx = 0; im_idx < num_images_; im_idx++) {
                // Jacobian matrix w.r.t. warping fields' params
                size_t num_params = warp_fields_[im_idx].GetNumParameters();
                Eigen::MatrixXd JTJ(num_params, num_params);
                Eigen::VectorXd JTr(num_params);
                JTJ.setZero();
                JTr.setZero();
                double residual = 0.0;
                size_t num_visible_pixels = 0;

                for (double u = 0; u < width_; u++) {
                    for (double v = 0; v < height_; v++) {
                        // if (!im_grays_[im_idx]->TestImageBoundary(u, v, 2)) {
                        //     continue;
                        // }
                        // Compute Jacobian and residual proxy value and pattern
                        int ii = (int)(u / anchor_step_);
                        int jj = (int)(v / anchor_step_);
                        if (ii >= anchor_w_ - 1 || jj >= anchor_h_ - 1) {
                            continue;
                        }
                        double p = (u - ii * anchor_step_) / anchor_step_;
                        double q = (v - jj * anchor_step_) / anchor_step_;
                        Eigen::Vector2d grids[4] = {
                                warp_fields_[im_idx].QueryFlow(ii, jj),
                                warp_fields_[im_idx].QueryFlow(ii, jj + 1),
                                warp_fields_[im_idx].QueryFlow(ii + 1, jj),
                                warp_fields_[im_idx].QueryFlow(ii + 1, jj + 1)};
                        Eigen::Vector2d uuvv = (1 - p) * (1 - q) * grids[0] +
                                               (1 - p) * (q)*grids[1] +
                                               (p) * (1 - q) * grids[2] +
                                               (p) * (q)*grids[3];
                        double uu = uuvv(0);
                        double vv = uuvv(1);
                        // std::cout << "(" << u << ", " << v << ") -> (" << uu
                        //           << ", " << vv << ")" << std::endl;
                        if (im_masks_[im_idx]->FloatValueAt(uu, vv).second !=
                            1) {
                            continue;
                        }
                        bool valid;
                        double im_gray_pixel_val, dIdfx, dIdfy;
                        std::tie(valid, im_gray_pixel_val) =
                                im_grays_[im_idx]->FloatValueAt(uu, vv);
                        std::tie(valid, dIdfx) =
                                im_dxs_[im_idx]->FloatValueAt(uu, vv);
                        std::tie(valid, dIdfy) =
                                im_dys_[im_idx]->FloatValueAt(uu, vv);
                        Eigen::Vector2d dIdf(dIdfx, dIdfy);
                        Eigen::Vector2d dfdx =
                                ((grids[2] - grids[0]) * (1 - q) +
                                 (grids[3] - grids[1]) * q) /
                                anchor_step_;
                        Eigen::Vector2d dfdy =
                                ((grids[1] - grids[0]) * (1 - p) +
                                 (grids[3] - grids[2]) * p) /
                                anchor_step_;

                        Eigen::Vector8d J_r;
                        J_r(0) = dIdf(0) * (1 - p) * (1 - q);
                        J_r(1) = dIdf(1) * (1 - p) * (1 - q);
                        J_r(2) = dIdf(0) * (1 - p) * (q);
                        J_r(3) = dIdf(1) * (1 - p) * (q);
                        J_r(4) = dIdf(0) * (p) * (1 - q);
                        J_r(5) = dIdf(1) * (p) * (1 - q);
                        J_r(6) = dIdf(0) * (p) * (q);
                        J_r(7) = dIdf(1) * (p) * (q);

                        Eigen::Vector8i pattern;
                        pattern(0) = (ii + jj * anchor_w_) * 2;
                        pattern(1) = (ii + jj * anchor_w_) * 2 + 1;
                        pattern(2) = (ii + (jj + 1) * anchor_w_) * 2;
                        pattern(3) = (ii + (jj + 1) * anchor_w_) * 2 + 1;
                        pattern(4) = ((ii + 1) + jj * anchor_w_) * 2;
                        pattern(5) = ((ii + 1) + jj * anchor_w_) * 2 + 1;
                        pattern(6) = ((ii + 1) + (jj + 1) * anchor_w_) * 2;
                        pattern(7) = ((ii + 1) + (jj + 1) * anchor_w_) * 2 + 1;

                        // Compute residual
                        double im_proxy_pixel_val;
                        std::tie(valid, im_proxy_pixel_val) =
                                im_proxy->FloatValueAt(uu, vv);
                        double r = im_gray_pixel_val - im_proxy_pixel_val;
                        residual += r * r;

                        // Accumulate to JTJ and JTr
                        for (auto x = 0; x < J_r.size(); x++) {
                            for (auto y = 0; y < J_r.size(); y++) {
                                JTJ(pattern(x), pattern(y)) += J_r(x) * J_r(y);
                            }
                        }
                        for (auto x = 0; x < J_r.size(); x++) {
                            JTr(pattern(x)) += r * J_r(x);
                        }

                        num_visible_pixels++;
                    }  // for (double v = 0; v < height_; v++)
                }      // for (double u = 0; u < width_; u++)

                // std::cout << "num_visible_pixels " << num_visible_pixels
                //           << std::endl;

                // Per image, update anchor point with weights
                double weight = option_.anchor_weight_ * num_visible_pixels /
                                width_ / height_;
                double residual_reg;
                for (int j = 0; j < num_params; j++) {
                    double r = weight * (warp_fields_[im_idx].flow_(j) -
                                         warp_fields_identity_.flow_(j));
                    JTJ(j, j) += weight * weight;
                    JTr(j) += weight * r;
                    residual_reg += r * r;
                }

                bool success;
                Eigen::VectorXd result;
                std::tie(success, result) = utility::SolveLinearSystemPSD(
                        JTJ, -JTr, /*prefer_sparse=*/false,
                        /*check_symmetric=*/false,
                        /*check_det=*/false, /*check_psd=*/false);
                for (int j = 0; j < num_params; j++) {
                    warp_fields_[im_idx].flow_(j) += result(j);
                }
#ifdef _OPENMP
#pragma omp critical
#endif
                {
                    residual_sum += residual;
                    residual_reg_sum += residual_reg;
                }
            }  // for (size_t im_idx = 0; im_idx < num_images_; im_idx++)

            // Update im_proxy after processing all images once
            im_proxy = ComputeWarpAverageImage();

            utility::PrintDebug("Residual error : %.6f, reg : %.6f\n",
                                residual_sum, residual_reg_sum);

        }  // for (size_t iter = 0; iter < num_iters; iter++)

    }  // void Optimize(size_t num_iters = 100)

    // Compute average image after warping
    std::shared_ptr<geometry::Image> ComputeWarpAverageImage() {
        std::vector<std::shared_ptr<geometry::Image>> im_warps(num_images_);
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
        for (size_t i = 0; i < num_images_; i++) {
            im_warps[i] = ComputeWarpedImage(*im_grays_[i], warp_fields_[i]);
        }
        return WarpFieldOptimizer::ComputeAverageImage(im_warps);
    }

protected:
    // Compute Warped image with image and warp field
    static std::shared_ptr<geometry::Image> ComputeWarpedImage(
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
                Eigen::Vector2d u_v_warp =
                        warp_field.GetImageWarpingField(u, v);
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

    // Compute average image
    static std::shared_ptr<geometry::Image> ComputeAverageImage(
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

public:
    std::vector<color_map::ImageWarpingField> warp_fields_;
    color_map::ImageWarpingField warp_fields_identity_;
    std::vector<std::shared_ptr<geometry::Image>> im_grays_;
    std::vector<std::shared_ptr<geometry::Image>> im_dxs_;  // dx of im_grays_
    std::vector<std::shared_ptr<geometry::Image>> im_dys_;  // dy of im_grays_
    std::vector<std::shared_ptr<geometry::Image>> im_masks_;

    int width_ = 0;
    int height_ = 0;
    int num_of_channels_ = 0;
    size_t num_images_ = 0;
    int anchor_w_;
    int anchor_h_;
    double anchor_step_;

    WarpFieldOptimizerOption option_;
};

std::pair<std::vector<std::shared_ptr<geometry::Image>>,
          std::vector<std::shared_ptr<geometry::Image>>>
ReadDataset(const std::string& root_dir,
            const std::string& im_pattern,
            const std::string& im_mask_pattern,
            int num_images) {
    std::vector<std::shared_ptr<geometry::Image>> im_grays;
    std::vector<std::shared_ptr<geometry::Image>> im_masks;
    for (int i = 0; i < num_images; i++) {
        // Get im_gray
        char buf[1000];
        int status =
                sprintf(buf, ("%s/" + im_pattern).c_str(), root_dir.c_str(), i);
        if (status < 0) {
            throw std::runtime_error("Image path formatting error.");
        }
        std::string im_path(buf);
        // std::cout << "Reading: " << im_path << std::endl;
        auto im_gray = std::make_shared<geometry::Image>();
        io::ReadImage(im_path, *im_gray);
        im_grays.push_back(im_gray->CreateFloatImage());

        // Get im_mask
        status = sprintf(buf, ("%s/" + im_mask_pattern).c_str(),
                         root_dir.c_str(), i);
        if (status < 0) {
            throw std::runtime_error("Image mask path formatting error.");
        }
        std::string im_mask_path(buf);
        // std::cout << "Reading: " << im_mask_path << std::endl;
        auto im_mask_rgb = std::make_shared<geometry::Image>();
        io::ReadImage(im_mask_path, *im_mask_rgb);
        auto im_mask = im_mask_rgb->CreateFloatImage();
        for (size_t u = 0; u < im_mask->width_; u++) {
            for (size_t v = 0; v < im_mask->height_; v++) {
                if (*im_mask->PointerAt<float>(u, v) != 0) {
                    *im_mask->PointerAt<float>(u, v) = 1;
                }
            }
        }
        im_masks.push_back(im_mask);
    }
    std::cout << "Read " << num_images << " images" << std::endl;
    return std::make_pair(im_grays, im_masks);
}

int main(int argc, char** args) {
    // Data path
    utility::SetVerbosityLevel(utility::VerbosityLevel::VerboseAlways);
    std::string im_dir = "/home/ylao/data/inverse-projection";
    std::cout << "im_dir: " << im_dir << std::endl;

    // Read images
    std::vector<std::shared_ptr<geometry::Image>> im_grays;
    std::vector<std::shared_ptr<geometry::Image>> im_masks;
    std::tie(im_grays, im_masks) = ReadDataset(im_dir, "delta-color-%d.png",
                                               "delta-weight-%d.png", 33);

    WarpFieldOptimizerOption option(/*iter*/ 50, /*v_anchors*/ 16,
                                    /*weight*/ 10);
    WarpFieldOptimizer wf_optimizer(im_grays, im_masks, option);

    auto im_warp_avg_init = wf_optimizer.ComputeWarpAverageImage();
    std::string im_warp_avg_init_path =
            im_dir + "/results/im_warp_avg_init.png";
    std::cout << "output im_warp_avg_init_path: " << im_warp_avg_init_path
              << std::endl;
    io::WriteImage(im_warp_avg_init_path,
                   *im_warp_avg_init->CreateImageFromFloatImage<uint8_t>());

    wf_optimizer.Optimize();

    auto im_warp_avg = wf_optimizer.ComputeWarpAverageImage();
    std::string im_warp_avg_path = im_dir + "/results/im_warp_avg.png";
    std::cout << "output im_warp_avg_path: " << im_warp_avg_path << std::endl;
    io::WriteImage(im_warp_avg_path,
                   *im_warp_avg->CreateImageFromFloatImage<uint8_t>());

    return 0;
}
