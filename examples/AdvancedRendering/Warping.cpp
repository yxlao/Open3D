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
#include <numeric>
#include <sstream>

#include "Open3D/Open3D.h"

using namespace open3d;

namespace Eigen {
typedef Eigen::Matrix<double, 8, 8> Matrix8d;
typedef Eigen::Matrix<double, 8, 1> Vector8d;
typedef Eigen::Matrix<int, 8, 1> Vector8i;
}  // namespace Eigen

template <typename T>
std::vector<size_t> sort_indexes(const std::vector<T>& v, bool ascend = true) {
    // https://stackoverflow.com/a/12399290/1255535
    // Initialize original index locations
    std::vector<size_t> idx(v.size());
    std::iota(idx.begin(), idx.end(), 0);

    // Sort indexes based on comparing values in v
    if (ascend) {
        std::sort(idx.begin(), idx.end(),
                  [&v](size_t i1, size_t i2) { return v[i1] < v[i2]; });
    } else {
        std::sort(idx.begin(), idx.end(),
                  [&v](size_t i1, size_t i2) { return v[i1] > v[i2]; });
    }
    return idx;
}

template <typename T>
std::vector<size_t> argmax_k(const std::vector<T>& v, size_t k) {
    k = std::min(k, v.size());
    std::vector<size_t> max_indices = sort_indexes(v, false);
    std::vector<size_t> k_max_indices(max_indices.begin(),
                                      max_indices.begin() + k);
    return k_max_indices;
}

class WarpFieldOptimizerOption {
public:
    WarpFieldOptimizerOption(
            // Attention: when you update the defaults, update the docstrings in
            // Python/color_map/color_map.cpp
            int num_iters = 100,
            int num_vertical_anchors = 10,
            double anchor_weight = 0.316,
            bool save_increments = false,
            const std::string& result_dir = "./inverse-proj-result")
        : num_iters_(num_iters),
          num_vertical_anchors_(num_vertical_anchors),
          anchor_weight_(anchor_weight),
          save_increments_(save_increments),
          result_dir_(result_dir) {}
    ~WarpFieldOptimizerOption() {}

public:
    int num_iters_;
    int num_vertical_anchors_;
    double anchor_weight_;
    bool save_increments_;
    std::string result_dir_;
};

class WarpFieldOptimizer {
public:
    WarpFieldOptimizer(
            const std::vector<std::shared_ptr<geometry::Image>>& im_rgbs,
            const std::vector<std::shared_ptr<geometry::Image>>& im_masks,
            const std::vector<std::shared_ptr<geometry::Image>>& im_weights,
            const std::shared_ptr<geometry::Image>& label,
            const WarpFieldOptimizerOption& option)
        : im_rgbs_(im_rgbs),
          im_masks_(im_masks),
          im_weights_(im_weights),
          im_label_(label),
          option_(option) {
        // TODO: ok to throw exception here?
        if (im_rgbs_.size() == 0) {
            throw std::runtime_error("Empty inputs");
        }

        // Prepare im_grays
        for (const auto& im_rgb : im_rgbs) {
            im_grays_.push_back(CreateFloatImageFromImage(*im_rgb));
        }

        // TODO: check that all images are of the same size
        width_ = im_grays_[0]->width_;
        height_ = im_grays_[0]->height_;
        num_of_channels_ = im_grays_[0]->num_of_channels_;
        num_images_ = im_grays_.size();

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
        for (const auto& im_gray : im_grays_) {
            im_dxs_.push_back(FilterImage(
                    *im_gray, geometry::Image::FilterType::Sobel3Dx));
            im_dys_.push_back(FilterImage(
                    *im_gray, geometry::Image::FilterType::Sobel3Dy));
        }

        // Conver to float im_r, im_g, im_b for color interpolation
        for (const auto& im_rgb : im_rgbs_) {
            auto im_r = std::make_shared<geometry::Image>();
            im_r->PrepareImage(width_, height_, 1, 4);
            auto im_g = std::make_shared<geometry::Image>();
            im_g->PrepareImage(width_, height_, 1, 4);
            auto im_b = std::make_shared<geometry::Image>();
            im_b->PrepareImage(width_, height_, 1, 4);

            for (int u = 0; u < width_; u++) {
                for (int v = 0; v < height_; v++) {
                    *(geometry::PointerAt<float>(*im_r, u, v)) = double(
                            *(geometry::PointerAt<uint8_t>(*im_rgb, u, v, 0)) /
                            255.);
                    *(geometry::PointerAt<float>(*im_g, u, v)) = double(
                            *(geometry::PointerAt<uint8_t>(*im_rgb, u, v, 1)) /
                            255.);
                    *(geometry::PointerAt<float>(*im_b, u, v)) = double(
                            *(geometry::PointerAt<uint8_t>(*im_rgb, u, v, 2)) /
                            255.);
                }
            }

            im_rs_.push_back(im_r);
            im_gs_.push_back(im_g);
            im_bs_.push_back(im_b);
        }
    }
    ~WarpFieldOptimizer() {}

    // Run optimization of warp_fields_
    void Optimize() {
        // Initialize proxy image with avg
        mask_proxy_ = ComputeInitMaskImage();

        std::shared_ptr<geometry::Image> im_proxy = ComputeWarpAverageImage();
        std::vector<std::shared_ptr<geometry::Image>> inverse_proxy_masks =
                ComputeInverseProxyMasks();

        for (size_t iter = 0; iter < option_.num_iters_; iter++) {
            double residual_sum = 0.0;
            double residual_reg_sum = 0.0;

            if (option_.save_increments_ && iter % 10 == 0) {
                auto im_avg = ComputeWarpAverageColorImage();

                std::stringstream im_path;
                im_path << option_.result_dir_ << "/" << std::setw(4)
                        << std::setfill('0') << iter << ".jpg";

                std::cout << "output im_warp_avg_init_path: " << im_path.str()
                          << std::endl;
                io::WriteImage(im_path.str(), *im_avg);
            }

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
                float sum_visible_pixels_weight = 0;

                for (double u = 0; u < width_; u++) {
                    for (double v = 0; v < height_; v++) {
                        if (*geometry::PointerAt<unsigned char>(*mask_proxy_, u,
                                                                v) == 0 ||
                            (*geometry::PointerAt<unsigned char>(
                                     *inverse_proxy_masks[im_idx], u, v) ==
                             0)) {
                            continue;
                        }

                        uint8_t label_proxy =
                                *geometry::PointerAt<uint8_t>(*im_label_, u, v);

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
                        uint8_t label_pixel = *geometry::PointerAt<uint8_t>(
                                *im_label_, (int)uu, (int)vv);
                        if (label_proxy != label_pixel) {
                            continue;
                        }

                        // Get the proxy reference value
                        bool valid;
                        double im_proxy_pixel_val;
                        std::tie(valid, im_proxy_pixel_val) =
                                im_proxy->FloatValueAt(u, v);

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
                        double r = (im_gray_pixel_val - im_proxy_pixel_val);
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

                        sum_visible_pixels_weight +=
                                im_weights_[im_idx]
                                        ->FloatValueAt(uu, vv)
                                        .second;
                    }  // for (double v = 0; v < height_; v++)
                }      // for (double u = 0; u < width_; u++)

                // std::cout << "num_visible_pixels " << num_visible_pixels
                //           << std::endl;

                // Per image, update anchor point with weights
                double weight = option_.anchor_weight_ *
                                sum_visible_pixels_weight / width_ / height_;
                double residual_reg = 0;
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
            inverse_proxy_masks = ComputeInverseProxyMasks();

            utility::PrintDebug("Residual error : %.6f, reg : %.6f\n",
                                residual_sum, residual_reg_sum);
        }  // for (size_t iter = 0; iter < num_iters; iter++)

    }  // void Optimize(size_t num_iters = 100)

    // Compute average image after warping
    std::shared_ptr<geometry::Image> ComputeWarpAverageImage() {
        std::vector<std::shared_ptr<geometry::Image>> inverse_proxy_masks =
                ComputeInverseProxyMasks();

        auto im_avg = std::make_shared<geometry::Image>();
        im_avg->PrepareImage(width_, height_, num_of_channels_, 4);

#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
        for (int u = 0; u < width_; u++) {
            for (int v = 0; v < height_; v++) {
                if (*geometry::PointerAt<unsigned char>(*mask_proxy_, u, v) ==
                    0) {
                    continue;
                }

                double pixel_val = 0;
                size_t num_visible_image = 0;
                for (size_t im_idx = 0; im_idx < num_images_; im_idx++) {
                    Eigen::Vector2d uuvv =
                            warp_fields_[im_idx].GetImageWarpingField(u, v);
                    double uu = uuvv(0);
                    double vv = uuvv(1);
                    if (im_masks_[im_idx]->FloatValueAt(uu, vv).second != 1) {
                        continue;
                    }
                    if (*geometry::PointerAt<unsigned char>(
                                *inverse_proxy_masks[im_idx], u, v) == 0) {
                        continue;
                    }
                    pixel_val += im_grays_[im_idx]->FloatValueAt(uu, vv).second;
                    num_visible_image++;
                }
                if (num_visible_image > 0) {
                    pixel_val /= num_visible_image;
                } else {
                    pixel_val = 0;
                }
                *(geometry::PointerAt<float>(*im_avg, u, v)) = pixel_val;
            }
        }
        return im_avg;
    }

    std::shared_ptr<geometry::Image> ComputeInitMaskImage() {
        auto mask = std::make_shared<geometry::Image>();
        mask->PrepareImage(width_, height_, 1, 1);

        for (int u = 0; u < width_; u++) {
            for (int v = 0; v < height_; v++) {
                *(geometry::PointerAt<unsigned char>(*mask, u, v)) = false;
            }
        }

        for (int u = 0; u < width_; u++) {
            for (int v = 0; v < height_; v++) {
                for (size_t im_idx = 0; im_idx < num_images_; im_idx++) {
                    *(geometry::PointerAt<unsigned char>(*mask, u, v)) =
                            (*(geometry::PointerAt<unsigned char>(*mask, u,
                                                                  v)) ||
                             (im_masks_[im_idx]->FloatValueAt(u, v).second > 0))
                                    ? 255
                                    : 0;
                }
            }
        }
        return mask;
    }

    // Compute average image after warping
    struct Pixel {
        double weight;
        double r;
        double g;
        double b;
        int idx;
    };

    // inverse_proxy_masks[im_idx].ValueAt(u, v) == 1 iff
    // im[im_idx] is used to compute average color for pixel (u, v)
    std::vector<std::shared_ptr<geometry::Image>> ComputeInverseProxyMasks()
            const {
        std::vector<std::shared_ptr<geometry::Image>> inverse_proxy_masks(
                im_masks_.size());
        for (auto& inverse_proxy_mask : inverse_proxy_masks) {
            inverse_proxy_mask = std::make_shared<geometry::Image>();
            inverse_proxy_mask->PrepareImage(width_, height_, 1, 1);
        }

#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
        for (int u = 0; u < width_; u++) {
            for (int v = 0; v < height_; v++) {
                uint8_t label_proxy =
                        *geometry::PointerAt<uint8_t>(*im_label_, u, v);

                for (size_t im_idx = 0; im_idx < num_images_; im_idx++) {
                    *geometry::PointerAt<unsigned char>(
                            *inverse_proxy_masks[im_idx], u, v) = 0;
                }

                if (*geometry::PointerAt<unsigned char>(*mask_proxy_, u, v) ==
                    0) {
                    continue;
                }

                std::vector<double> candidate_weights;
                std::vector<size_t> candidate_indices;
                for (size_t im_idx = 0; im_idx < num_images_; im_idx++) {
                    Eigen::Vector2d uuvv =
                            warp_fields_[im_idx].GetImageWarpingField(u, v);
                    double uu = uuvv(0);
                    double vv = uuvv(1);

                    if (im_masks_[im_idx]->FloatValueAt(uu, vv).second < 0.5) {
                        continue;
                    }

                    uint8_t label_pixel = *geometry::PointerAt<uint8_t>(
                            *im_label_, (int)uu, (int)vv);
                    if (label_pixel != label_proxy) continue;

                    candidate_weights.push_back(
                            im_weights_[im_idx]->FloatValueAt(uu, vv).second);
                    candidate_indices.push_back(im_idx);
                }

                for (size_t i : argmax_k(candidate_weights, 5)) {
                    *geometry::PointerAt<unsigned char>(
                            *inverse_proxy_masks[candidate_indices[i]], u, v) =
                            1;
                }
            }
        }
        return inverse_proxy_masks;
    }

    std::shared_ptr<geometry::Image> ComputeWarpAverageColorImage() const {
        std::vector<std::shared_ptr<geometry::Image>> inverse_proxy_masks =
                ComputeInverseProxyMasks();

        auto im_avg = std::make_shared<geometry::Image>();
        im_avg->PrepareImage(width_, height_, 3, 1);

#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
        for (int u = 0; u < width_; u++) {
            for (int v = 0; v < height_; v++) {
                double r_sum = 0;
                double g_sum = 0;
                double b_sum = 0;
                double weight_sum = 0;

                for (size_t im_idx = 0; im_idx < num_images_; im_idx++) {
                    Eigen::Vector2d uuvv =
                            warp_fields_[im_idx].GetImageWarpingField(u, v);
                    double uu = uuvv(0);
                    double vv = uuvv(1);
                    if (*geometry::PointerAt<unsigned char>(
                                *inverse_proxy_masks[im_idx], u, v) == 1) {
                        double weight = im_weights_[im_idx]
                                                ->FloatValueAt(uu, vv)
                                                .second;
                        r_sum += im_rs_[im_idx]->FloatValueAt(uu, vv).second *
                                 weight;
                        g_sum += im_gs_[im_idx]->FloatValueAt(uu, vv).second *
                                 weight;
                        b_sum += im_bs_[im_idx]->FloatValueAt(uu, vv).second *
                                 weight;
                        weight_sum += weight;
                    }
                }

                double r = 0;
                double g = 0;
                double b = 0;
                if (weight_sum > 0) {
                    r = r_sum / weight_sum;
                    g = g_sum / weight_sum;
                    b = b_sum / weight_sum;
                }

                *(geometry::PointerAt<uint8_t>(*im_avg, u, v, 0)) =
                        static_cast<uint8_t>(r * 255.);
                *(geometry::PointerAt<uint8_t>(*im_avg, u, v, 1)) =
                        static_cast<uint8_t>(g * 255.);
                *(geometry::PointerAt<uint8_t>(*im_avg, u, v, 2)) =
                        static_cast<uint8_t>(b * 255.);
            }
        }
        return im_avg;
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
        im_warped->PrepareImage(width, height, num_of_channels,
                                bytes_per_channel);

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
                    *(geometry::PointerAt<float>(*im_warped, u, v)) = pixel_val;
                } else {
                    *(geometry::PointerAt<float>(*im_warped, u, v)) = 0;
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
        im_avg->PrepareImage(width, height, num_of_channels, 4);
        for (int u = 0; u < width; ++u) {
            for (int v = 0; v < height; ++v) {
                *(geometry::PointerAt<float>(*im_avg, u, v)) = 0;
                for (const auto& im_gray : im_grays) {
                    *(geometry::PointerAt<float>(*im_avg, u, v)) +=
                            *(geometry::PointerAt<float>(*im_gray, u, v));
                }
                *(geometry::PointerAt<float>(*im_avg, u, v)) /= num_images;
            }
        }
        return im_avg;
    }

public:
    std::vector<color_map::ImageWarpingField> warp_fields_;
    color_map::ImageWarpingField warp_fields_identity_;
    std::vector<std::shared_ptr<geometry::Image>> im_rgbs_;
    std::vector<std::shared_ptr<geometry::Image>> im_rs_;
    std::vector<std::shared_ptr<geometry::Image>> im_gs_;
    std::vector<std::shared_ptr<geometry::Image>> im_bs_;
    std::vector<std::shared_ptr<geometry::Image>> im_grays_;
    std::vector<std::shared_ptr<geometry::Image>> im_dxs_;  // dx of im_grays_
    std::vector<std::shared_ptr<geometry::Image>> im_dys_;  // dy of im_grays_
    std::vector<std::shared_ptr<geometry::Image>> im_masks_;
    std::vector<std::shared_ptr<geometry::Image>> im_weights_;

    std::shared_ptr<geometry::Image> mask_proxy_;
    std::shared_ptr<geometry::Image> im_label_;

    int width_ = 0;
    int height_ = 0;
    int num_of_channels_ = 0;
    size_t num_images_ = 0;
    int anchor_w_;
    int anchor_h_;
    double anchor_step_;

    WarpFieldOptimizerOption option_;
};

std::tuple<std::vector<std::shared_ptr<geometry::Image>>,
           std::vector<std::shared_ptr<geometry::Image>>,
           std::vector<std::shared_ptr<geometry::Image>>>
ReadDataset(const std::string& root_dir,
            const std::string& im_pattern,
            const std::string& im_mask_pattern,
            int num_images) {
    std::vector<std::shared_ptr<geometry::Image>> im_rgbs;
    std::vector<std::shared_ptr<geometry::Image>> im_masks;
    std::vector<std::shared_ptr<geometry::Image>> im_weights;
    for (int i = 0; i < num_images; i++) {
        // Get im_rgb
        char buf[1000];
        int status =
                sprintf(buf, ("%s/" + im_pattern).c_str(), root_dir.c_str(), i);
        if (status < 0) {
            throw std::runtime_error("Image path formatting error.");
        }
        std::string im_path(buf);
        std::cout << "Reading: " << im_path << std::endl;
        auto im_rgb = std::make_shared<geometry::Image>();
        io::ReadImage(im_path, *im_rgb);
        im_rgbs.push_back(im_rgb);

        // Get im_mask
        status = sprintf(buf, ("%s/" + im_mask_pattern).c_str(),
                         root_dir.c_str(), i);
        if (status < 0) {
            throw std::runtime_error("Image mask path formatting error.");
        }
        std::string im_mask_path(buf);
        std::cout << "Reading: " << im_mask_path << std::endl;
        auto im_mask_rgb = std::make_shared<geometry::Image>();
        io::ReadImage(im_mask_path, *im_mask_rgb);

        // (0, weight, mask)
        auto im_mask = std::make_shared<geometry::Image>();
        auto im_weight = std::make_shared<geometry::Image>();
        im_mask->PrepareImage(im_mask_rgb->width_, im_mask_rgb->height_, 1, 4);
        im_weight->PrepareImage(im_mask_rgb->width_, im_mask_rgb->height_, 1,
                                4);
        for (size_t u = 0; u < im_mask->width_; u++) {
            for (size_t v = 0; v < im_mask->height_; v++) {
                *geometry::PointerAt<float>(*im_mask, u, v) =
                        *geometry::PointerAt<unsigned char>(*im_mask_rgb, u, v,
                                                            2) > 0
                                ? 1
                                : 0;
                *geometry::PointerAt<float>(*im_weight, u, v) =
                        *geometry::PointerAt<unsigned char>(*im_mask_rgb, u, v,
                                                            1) /
                        255.0f;

                //                printf("mask = %f, weight = %f\n",
                //                       *geometry::PointerAt<float>(*im_mask,
                //                       u, v),
                //                       *geometry::PointerAt<float>(*im_weight,
                //                       u, v));
            }
        }
        im_masks.push_back(im_mask);
        im_weights.push_back(im_weight);
    }

    std::cout << "Read " << num_images << " images" << std::endl;
    return std::make_tuple(im_rgbs, im_masks, im_weights);
}

int main(int argc, char** argv) {
    // Data path
    utility::SetVerbosityLevel(utility::VerbosityLevel::VerboseAlways);
    std::string im_dir = "inverse-proj-data";
    std::string res_dir = "inverse-proj-result";
    if (argc == 2) {
        im_dir = std::string(argv[1]) + "/" + im_dir;
        res_dir = std::string(argv[1]) + "/" + res_dir;
    }

    std::cout << "im_dir: " << im_dir << std::endl;

    std::vector<int> results(256);
    std::fill(results.begin(), results.end(), 0);
    auto labels = io::CreateImageFromFile(im_dir + "/labels.png");
    for (int i = 0; i < labels->width_; ++i) {
        for (int j = 0; j < labels->height_; ++j) {
            results[*geometry::PointerAt<uint8_t>(*labels, i, j)]++;
        }
    }
    for (int i = 0; i < results.size(); ++i) {
        printf("label[%d] = %d\n", i, results[i]);
    }

    auto im = std::make_shared<geometry::Image>();

    // Read images
    std::vector<std::shared_ptr<geometry::Image>> im_rgbs;
    std::vector<std::shared_ptr<geometry::Image>> im_masks;
    std::vector<std::shared_ptr<geometry::Image>> im_weights;
    std::tie(im_rgbs, im_masks, im_weights) = ReadDataset(
            im_dir, "delta-color-%d.png", "delta-weight-%d.png", 33);

    WarpFieldOptimizerOption option(/*iter*/ 500, /*v_anchors*/ 25,
                                    /*weight*/ 0.3,
                                    /* save_increments_ */ true,
                                    /* result_dir */ res_dir);
    WarpFieldOptimizer wf_optimizer(im_rgbs, im_masks, im_weights, labels,
                                    option);

    auto im_mask = wf_optimizer.ComputeInitMaskImage();
    std::string im_mask_path = res_dir + "/im_init_mask.png";
    std::cout << "output im_init_mask path: " << im_mask_path << std::endl;
    io::WriteImage(im_mask_path, *im_mask);

    wf_optimizer.Optimize();

    auto im_warp_avg = wf_optimizer.ComputeWarpAverageColorImage();
    std::string im_warp_avg_path = res_dir + "/im_warp_avg.png";
    std::cout << "output im_warp_avg_path: " << im_warp_avg_path << std::endl;
    io::WriteImage(im_warp_avg_path, *im_warp_avg);

    return 0;
}
