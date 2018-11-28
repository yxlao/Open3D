//
// Created by wei on 11/14/18.
//

//
// Created by wei on 10/6/18.
//

#include <string>
#include <vector>
#include <Core/Core.h>
#include <IO/IO.h>
#include <Cuda/Odometry/RGBDOdometryCuda.h>
#include <Cuda/Integration/ScalableTSDFVolumeCuda.h>
#include <Cuda/Integration/ScalableMeshVolumeCuda.h>
#include <Cuda/Geometry/PointCloudCuda.h>
#include <Visualization/Visualization.h>

#include <opencv2/opencv.hpp>
#include <thread>

using namespace open3d;

std::tuple<std::shared_ptr<Image>, std::shared_ptr<Image>>
InitializeCorrespondenceMap(int width, int height)
{
    // initialization: filling with any (u,v) to (-1,-1)
    auto correspondence_map = std::make_shared<Image>();
    auto depth_buffer = std::make_shared<Image>();
    correspondence_map->PrepareImage(width, height, 2, 4);
    depth_buffer->PrepareImage(width, height, 1, 4);
    for (int v = 0; v < correspondence_map->height_; v++) {
        for (int u = 0; u < correspondence_map->width_; u++) {
            *PointerAt<int>(*correspondence_map, u, v, 0) = -1;
            *PointerAt<int>(*correspondence_map, u, v, 1) = -1;
            *PointerAt<float>(*depth_buffer, u, v, 0) = -1.0f;
        }
    }
    return std::make_tuple(correspondence_map, depth_buffer);
}

inline void AddElementToCorrespondenceMap(
    Image &correspondence_map, Image &depth_buffer,
    int u_s, int v_s, int u_t, int v_t, float transformed_d_t) {
    int exist_u_t, exist_v_t;
    double exist_d_t;
    exist_u_t = *PointerAt<int>(correspondence_map, u_s, v_s, 0);
    exist_v_t = *PointerAt<int>(correspondence_map, u_s, v_s, 1);
    if (exist_u_t != -1 && exist_v_t != -1) {
        exist_d_t = *PointerAt<float>(depth_buffer, u_s, v_s);
        if (transformed_d_t < exist_d_t) { // update nearer point as correspondence
            *PointerAt<int>(correspondence_map, u_s, v_s, 0) = u_t;
            *PointerAt<int>(correspondence_map, u_s, v_s, 1) = v_t;
            *PointerAt<float>(depth_buffer, u_s, v_s) = transformed_d_t;
        }
    } else { // register correspondence
        *PointerAt<int>(correspondence_map, u_s, v_s, 0) = u_t;
        *PointerAt<int>(correspondence_map, u_s, v_s, 1) = v_t;
        *PointerAt<float>(depth_buffer, u_s, v_s) = transformed_d_t;
    }
}

void MergeCorrespondenceMaps(
    Image &correspondence_map, Image &depth_buffer,
    Image &correspondence_map_part, Image &depth_buffer_part)
{
    for (int v_s = 0; v_s < correspondence_map.height_; v_s++) {
        for (int u_s = 0; u_s < correspondence_map.width_; u_s++) {
            int u_t = *PointerAt<int>(correspondence_map_part, u_s, v_s, 0);
            int v_t = *PointerAt<int>(correspondence_map_part, u_s, v_s, 1);
            if (u_t != -1 && v_t != -1) {
                float transformed_d_t = *PointerAt<float>(
                    depth_buffer_part, u_s, v_s);
                AddElementToCorrespondenceMap(correspondence_map,
                                              depth_buffer, u_s, v_s, u_t, v_t, transformed_d_t);
            }
        }
    }
}

int CountCorrespondence(const Image &correspondence_map)
{
    int correspondence_count = 0;
    for (int v_s = 0; v_s < correspondence_map.height_; v_s++) {
        for (int u_s = 0; u_s < correspondence_map.width_; u_s++) {
            int u_t = *PointerAt<int>(correspondence_map, u_s, v_s, 0);
            int v_t = *PointerAt<int>(correspondence_map, u_s, v_s, 1);
            if (u_t != -1 && v_t != -1) {
                correspondence_count++;
            }
        }
    }
    return correspondence_count;
}

std::shared_ptr<CorrespondenceSetPixelWise> ComputeCorrespondence(
    const Eigen::Matrix3d intrinsic_matrix,
    const Eigen::Matrix4d &extrinsic,
    const Image &depth_s, const Image &depth_t,
    const OdometryOption &option)
{
    const Eigen::Matrix3d K = intrinsic_matrix;
    const Eigen::Matrix3d K_inv = K.inverse();
    const Eigen::Matrix3d R = extrinsic.block<3, 3>(0, 0);
    const Eigen::Matrix3d KRK_inv = K * R * K_inv;
    Eigen::Vector3d Kt = K * extrinsic.block<3, 1>(0, 3);

    std::shared_ptr<Image> correspondence_map;
    std::shared_ptr<Image> depth_buffer;
    std::tie(correspondence_map, depth_buffer) =
        InitializeCorrespondenceMap(depth_t.width_, depth_t.height_);

#ifdef _OPENMP
#pragma omp parallel
    {
#endif
        std::shared_ptr<Image> correspondence_map_private;
        std::shared_ptr<Image> depth_buffer_private;
        std::tie(correspondence_map_private, depth_buffer_private) =
            InitializeCorrespondenceMap(depth_t.width_, depth_t.height_);
#ifdef _OPENMP
#pragma omp for nowait
#endif
        for (int v_s = 0; v_s < depth_s.height_; v_s++) {
            for (int u_s = 0; u_s < depth_s.width_; u_s++) {
                double d_s = *PointerAt<float>(depth_s, u_s, v_s);
                if (!std::isnan(d_s)) {
                    Eigen::Vector3d uv_in_s =
                        d_s * KRK_inv * Eigen::Vector3d(u_s, v_s, 1.0) + Kt;
                    double transformed_d_s = uv_in_s(2);
                    int u_t = (int)(uv_in_s(0) / transformed_d_s + 0.5);
                    int v_t = (int)(uv_in_s(1) / transformed_d_s + 0.5);
                    if (u_t >= 0 && u_t < depth_t.width_ &&
                        v_t >= 0 && v_t < depth_t.height_) {
                        double d_t = *PointerAt<float>(depth_t, u_t, v_t);
                        if (!std::isnan(d_t) && std::abs(transformed_d_s - d_t)
                            <= option.max_depth_diff_) {
                            AddElementToCorrespondenceMap(
                                *correspondence_map_private,
                                *depth_buffer_private,
                                u_s, v_s, u_t, v_t, (float)d_s);
                        }
                    }
                }
            }
        }
#ifdef _OPENMP
#pragma omp critical
        {
#endif
            MergeCorrespondenceMaps(
                *correspondence_map, *depth_buffer,
                *correspondence_map_private, *depth_buffer_private);
#ifdef _OPENMP
        }    //    omp critical
    }    //    omp parallel
#endif

    auto correspondence = std::make_shared<CorrespondenceSetPixelWise>();
    int correspondence_count = CountCorrespondence(*correspondence_map);
    correspondence->resize(correspondence_count);
    int cnt = 0;
    for (int v_s = 0; v_s < correspondence_map->height_; v_s++) {
        for (int u_s = 0; u_s < correspondence_map->width_; u_s++) {
            int u_t = *PointerAt<int>(*correspondence_map, u_s, v_s, 0);
            int v_t = *PointerAt<int>(*correspondence_map, u_s, v_s, 1);
            if (u_t != -1 && v_t != -1) {
                Eigen::Vector4i pixel_correspondence(u_s, v_s, u_t, v_t);
                (*correspondence)[cnt] = pixel_correspondence;
                cnt++;
            }
        }
    }
    return correspondence;
}

std::shared_ptr<Image> ConvertDepthImageToXYZImage(
    const Image &depth, const Eigen::Matrix3d &intrinsic_matrix)
{
    auto image_xyz = std::make_shared<Image>();
    if (depth.num_of_channels_ != 1 || depth.bytes_per_channel_ != 4) {
        PrintDebug("[ConvertDepthImageToXYZImage] Unsupported image format.\n");
        return image_xyz;
    }
    const double inv_fx = 1.0 / intrinsic_matrix(0, 0);
    const double inv_fy = 1.0 / intrinsic_matrix(1, 1);
    const double ox = intrinsic_matrix(0, 2);
    const double oy = intrinsic_matrix(1, 2);
    image_xyz->PrepareImage(depth.width_, depth.height_, 3, 4);

    for (int y = 0; y < image_xyz->height_; y++) {
        for (int x = 0; x < image_xyz->width_; x++) {
            float *px = PointerAt<float>(*image_xyz, x, y, 0);
            float *py = PointerAt<float>(*image_xyz, x, y, 1);
            float *pz = PointerAt<float>(*image_xyz, x, y, 2);
            float z = *PointerAt<float>(depth, x, y);
            *px = (float)((x - ox) * z * inv_fx);
            *py = (float)((y - oy) * z * inv_fy);
            *pz = z;
        }
    }
    return image_xyz;
}

std::vector<Eigen::Matrix3d>
CreateCameraMatrixPyramid(
    const PinholeCameraIntrinsic &pinhole_camera_intrinsic, int levels)
{
    std::vector<Eigen::Matrix3d> pyramid_camera_matrix;
    pyramid_camera_matrix.reserve(levels);
    for (int i = 0; i < levels; i++) {
        Eigen::Matrix3d level_camera_matrix;
        if (i == 0)
            level_camera_matrix = pinhole_camera_intrinsic.intrinsic_matrix_;
        else
            level_camera_matrix = 0.5 * pyramid_camera_matrix[i - 1];
        level_camera_matrix(2, 2) = 1.;
        pyramid_camera_matrix.push_back(level_camera_matrix);
    }
    return pyramid_camera_matrix;
}

Eigen::Matrix6d CreateInformationMatrix(
    const Eigen::Matrix4d &extrinsic,
    const PinholeCameraIntrinsic &pinhole_camera_intrinsic,
    const Image &depth_s, const Image &depth_t,
    const OdometryOption &option)
{
    auto correspondence = ComputeCorrespondence(
        pinhole_camera_intrinsic.intrinsic_matrix_,
        extrinsic, depth_s, depth_t, option);

    auto xyz_t = ConvertDepthImageToXYZImage(
        depth_t, pinhole_camera_intrinsic.intrinsic_matrix_);

    // write q^*
    // see http://redwood-data.org/indoor/registration.html
    // note: I comes first and q_skew is scaled by factor 2.
    Eigen::Matrix6d GTG = Eigen::Matrix6d::Identity();
#ifdef _OPENMP
#pragma omp parallel
    {
#endif
        Eigen::Matrix6d GTG_private = Eigen::Matrix6d::Identity();
        Eigen::Vector6d G_r_private = Eigen::Vector6d::Zero();
#ifdef _OPENMP
#pragma omp for nowait
#endif
        for (auto row = 0; row < correspondence->size(); row++) {
            int u_t = (*correspondence)[row](2);
            int v_t = (*correspondence)[row](3);
            double x = *PointerAt<float>(*xyz_t, u_t, v_t, 0);
            double y = *PointerAt<float>(*xyz_t, u_t, v_t, 1);
            double z = *PointerAt<float>(*xyz_t, u_t, v_t, 2);
            G_r_private.setZero();
            G_r_private(1) = z;
            G_r_private(2) = -y;
            G_r_private(3) = 1.0;
            GTG_private.noalias() += G_r_private * G_r_private.transpose();
            G_r_private.setZero();
            G_r_private(0) = -z;
            G_r_private(2) = x;
            G_r_private(4) = 1.0;
            GTG_private.noalias() += G_r_private * G_r_private.transpose();
            G_r_private.setZero();
            G_r_private(0) = y;
            G_r_private(1) = -x;
            G_r_private(5) = 1.0;
            GTG_private.noalias() += G_r_private * G_r_private.transpose();
        }
#ifdef _OPENMP
#pragma omp critical
#endif
        {
            GTG += GTG_private;
        }
#ifdef _OPENMP
    }
#endif
    return std::move(GTG);
}

void NormalizeIntensity(Image &image_s, Image &image_t,
                        CorrespondenceSetPixelWise &correspondence)
{
    if (image_s.width_ != image_t.width_ ||
        image_s.height_ != image_t.height_) {
        PrintError("[NormalizeIntensity] Size of two input images should be same\n");
        return;
    }
    double mean_s = 0.0, mean_t = 0.0;
    for (int row = 0; row < correspondence.size(); row++) {
        int u_s = correspondence[row](0);
        int v_s = correspondence[row](1);
        int u_t = correspondence[row](2);
        int v_t = correspondence[row](3);
        mean_s += *PointerAt<float>(image_s, u_s, v_s);
        mean_t += *PointerAt<float>(image_t, u_t, v_t);
    }
    mean_s /= (double)correspondence.size();
    mean_t /= (double)correspondence.size();
    LinearTransformImage(image_s, 0.5 / mean_s, 0.0);
    LinearTransformImage(image_t, 0.5 / mean_t, 0.0);
}

inline std::shared_ptr<RGBDImage> PackRGBDImage(
    const Image &color, const Image &depth) {
    return std::make_shared<RGBDImage>(RGBDImage(color, depth));
}

std::shared_ptr<Image> PreprocessDepth(
    const Image &depth_orig, const OdometryOption &option)
{
    std::shared_ptr<Image> depth_processed = std::make_shared<Image>();
    *depth_processed = depth_orig;
    for (int y = 0; y < depth_processed->height_; y++) {
        for (int x = 0; x < depth_processed->width_; x++) {
            float *p = PointerAt<float>(*depth_processed, x, y);
            if ((*p < option.min_depth_ || *p > option.max_depth_ || *p <= 0))
                *p = std::numeric_limits<float>::quiet_NaN();
        }
    }
    return depth_processed;
}

inline bool CheckImagePair(const Image &image_s, const Image &image_t)
{
    return (image_s.width_ == image_t.width_ &&
        image_s.height_ == image_t.height_);
}

inline bool CheckRGBDImagePair(const RGBDImage &source, const RGBDImage &target)
{
    return (CheckImagePair(source.color_, target.color_) &&
        CheckImagePair(source.depth_, target.depth_) &&
        CheckImagePair(source.color_, source.depth_) &&
        CheckImagePair(target.color_, target.depth_) &&
        CheckImagePair(source.color_, target.color_) &&
        source.color_.num_of_channels_ == 1 &&
        source.depth_.num_of_channels_ == 1 &&
        target.color_.num_of_channels_ == 1 &&
        target.depth_.num_of_channels_ == 1 &&
        source.color_.bytes_per_channel_ == 4 &&
        target.color_.bytes_per_channel_ == 4 &&
        source.depth_.bytes_per_channel_ == 4 &&
        target.depth_.bytes_per_channel_ == 4);
}

std::tuple<std::shared_ptr<RGBDImage>, std::shared_ptr<RGBDImage>>
InitializeRGBDOdometry(
    const RGBDImage &source, const RGBDImage &target,
    const PinholeCameraIntrinsic &pinhole_camera_intrinsic,
    const Eigen::Matrix4d &odo_init,
    const OdometryOption &option)
{
    auto source_gray = FilterImage(source.color_,
                                   Image::FilterType::Gaussian3);
    auto target_gray = FilterImage(target.color_,
                                   Image::FilterType::Gaussian3);
    auto source_depth_preprocessed = PreprocessDepth(source.depth_, option);
    auto target_depth_preprocessed = PreprocessDepth(target.depth_, option);
    auto source_depth = FilterImage(*source_depth_preprocessed,
                                    Image::FilterType::Gaussian3);
    auto target_depth = FilterImage(*target_depth_preprocessed,
                                    Image::FilterType::Gaussian3);

    auto correspondence = ComputeCorrespondence(
        pinhole_camera_intrinsic.intrinsic_matrix_, odo_init,
        *source_depth, *target_depth, option);
    NormalizeIntensity(*source_gray, *target_gray, *correspondence);

    auto source_out = PackRGBDImage(*source_gray, *source_depth);
    auto target_out = PackRGBDImage(*target_gray, *target_depth);
    return std::make_tuple(source_out, target_out);
}

std::tuple<bool, Eigen::Matrix4d> DoSingleIteration(
    int iter, int level,
    const RGBDImage &source, const RGBDImage &target,
    const Image &source_xyz,
    const RGBDImage &target_dx, const RGBDImage &target_dy,
    const Eigen::Matrix3d intrinsic,
    const Eigen::Matrix4d &extrinsic_initial,
    const RGBDOdometryJacobian &jacobian_method,
    const OdometryOption &option)
{
    auto correspondence = ComputeCorrespondence(
        intrinsic, extrinsic_initial, source.depth_, target.depth_, option);
    int corresps_count = (int)correspondence->size();

    auto f_lambda = [&]
        (int i, std::vector<Eigen::Vector6d> &J_r, std::vector<double> &r) {
        jacobian_method.ComputeJacobianAndResidual(i, J_r, r,
                                                   source, target, source_xyz, target_dx, target_dy,
                                                   intrinsic, extrinsic_initial, *correspondence);
    };
    PrintDebug("Iter : %d, Level : %d, ", iter, level);
    Eigen::Matrix6d JTJ;
    Eigen::Vector6d JTr;
    std::tie(JTJ, JTr) = ComputeJTJandJTr<Eigen::Matrix6d, Eigen::Vector6d>(
        f_lambda, corresps_count);

    bool is_success;
    Eigen::Matrix4d extrinsic;
    std::tie(is_success, extrinsic) =
        SolveJacobianSystemAndObtainExtrinsicMatrix(JTJ, JTr);
    if (!is_success) {
        PrintWarning("[ComputeOdometry] no solution!\n");
        return std::make_tuple(false, Eigen::Matrix4d::Identity());
    } else {
        return std::make_tuple(true, extrinsic);
    }
}

std::tuple<bool, Eigen::Matrix4d> ComputeMultiscale(
    const RGBDImage &source, const RGBDImage &target,
    const PinholeCameraIntrinsic &pinhole_camera_intrinsic,
    const Eigen::Matrix4d &extrinsic_initial,
    const RGBDOdometryJacobian &jacobian_method,
    const OdometryOption &option)
{
    std::vector<int> iter_counts = option.iteration_number_per_pyramid_level_;
    int num_levels = (int)iter_counts.size();

    auto source_pyramid = CreateRGBDImagePyramid(source, num_levels);
    auto target_pyramid = CreateRGBDImagePyramid(target, num_levels);
    auto target_pyramid_dx = FilterRGBDImagePyramid
        (target_pyramid, Image::FilterType::Sobel3Dx);
    auto target_pyramid_dy = FilterRGBDImagePyramid
        (target_pyramid, Image::FilterType::Sobel3Dy);

    Eigen::Matrix4d result_odo = extrinsic_initial.isZero() ?
                                 Eigen::Matrix4d::Identity() : extrinsic_initial;

    std::vector<Eigen::Matrix3d> pyramid_camera_matrix =
        CreateCameraMatrixPyramid(pinhole_camera_intrinsic,
                                  (int)iter_counts.size());

    VisualizerWithKeyCallback visualizer;
    if (!visualizer.CreateVisualizerWindow("ScalableFusion", 640, 480, 0, 0)) {
        PrintWarning("Failed creating OpenGL window.\n");
    }

    visualizer.BuildUtilities();
    visualizer.UpdateWindowTitle();

    const int kIterations[3] = {60, 60, 60};
    bool finished = false;
    int level = 2;
    int iter = kIterations[level];
    Eigen::Matrix4d prev_transform = Eigen::Matrix4d::Identity();

    auto pcl_source = CreatePointCloudFromRGBDImage(source,pinhole_camera_intrinsic);
    auto pcl_target = CreatePointCloudFromRGBDImage(target,pinhole_camera_intrinsic);
    auto lines = std::make_shared<LineSet>();

    visualizer.AddGeometry(pcl_source);
    visualizer.AddGeometry(pcl_target);
    visualizer.AddGeometry(lines);

    visualizer.RegisterKeyCallback(GLFW_KEY_SPACE, [&](Visualizer *vis) {
        if (finished) return false;

        const Eigen::Matrix3d level_camera_matrix = pyramid_camera_matrix[level];

        auto source_xyz_level = ConvertDepthImageToXYZImage(
            source_pyramid[level]->depth_, level_camera_matrix);
        auto source_level = PackRGBDImage(source_pyramid[level]->color_,
                                          source_pyramid[level]->depth_);
        auto target_level = PackRGBDImage(target_pyramid[level]->color_,
                                          target_pyramid[level]->depth_);
        auto target_dx_level = PackRGBDImage(target_pyramid_dx[level]->color_,
                                             target_pyramid_dx[level]->depth_);
        auto target_dy_level = PackRGBDImage(target_pyramid_dy[level]->color_,
                                             target_pyramid_dy[level]->depth_);

        Eigen::Matrix4d curr_odo;
        bool is_success;

        auto correspondence = ComputeCorrespondence(
            level_camera_matrix, extrinsic_initial,
            source_level->depth_, target_level->depth_, option);

        std::tie(is_success, curr_odo) = DoSingleIteration(
            iter, level,
            *source_level, *target_level, *source_xyz_level,
            *target_dx_level, *target_dy_level, level_camera_matrix,
            result_odo, jacobian_method, option);

        result_odo = curr_odo * result_odo;

        --iter;
        if (iter == 0) {
            --level;
            if (level < 0) {
                finished = true;
            } else {
                iter = kIterations[level];
            }
        }

        float factor = 1.0f / (1 << level);
        PinholeCameraIntrinsic intrinsic(
            pinhole_camera_intrinsic.width_ * factor,
            pinhole_camera_intrinsic.height_ * factor,
            pinhole_camera_intrinsic.intrinsic_matrix_(0, 0) * factor,
            pinhole_camera_intrinsic.intrinsic_matrix_(1, 1) * factor,
            pinhole_camera_intrinsic.intrinsic_matrix_(0, 2) * factor,
            pinhole_camera_intrinsic.intrinsic_matrix_(1, 2) * factor);

        auto pcl_source_tmp = CreatePointCloudFromRGBDImage(*source_level,
            intrinsic, result_odo.inverse());
        auto pcl_target_tmp = CreatePointCloudFromRGBDImage(*target_level,
            intrinsic);

        pcl_source->points_ = pcl_source_tmp->points_;
        pcl_source->colors_.resize(pcl_source->points_.size());
        std::fill(pcl_source->colors_.begin(), pcl_source->colors_.end(),
            Eigen::Vector3d(0, 1, 0));
        pcl_target->points_ = pcl_target_tmp->points_;
        pcl_target->colors_.resize(pcl_target->points_.size());
        std::fill(pcl_target->colors_.begin(), pcl_target->colors_.end(),
                  Eigen::Vector3d(1, 0, 0));

        lines->points_.clear();
        lines->lines_.clear();
        lines->colors_.clear();
        auto correspondences = *correspondence;
        Eigen::Matrix3d Kinv = level_camera_matrix.inverse();
        for (int i = 0; i < correspondences.size(); ++i) {
            auto &c = correspondences[i];

            Eigen::Vector3d X_src =
                Kinv * Eigen::Vector3d(c(0), c(1), 1) * (*PointerAt<float>
                    (source_level->depth_, c(0), c(1)));
            Eigen::Vector4d X_src_h = result_odo * Eigen::Vector4d(X_src(0),X_src(1), X_src(2), 1.0);
            lines->points_.emplace_back(X_src_h.hnormalized());

            Eigen::Vector3d X_tgt =
                Kinv * Eigen::Vector3d(c(2), c(3), 1) * (*PointerAt<float>
                    (target_level->depth_, c(2), c(3)));
            lines->points_.emplace_back(X_tgt(0), X_tgt(1), X_tgt(2));

            lines->colors_.emplace_back(0, 0, 1);
            lines->lines_.emplace_back(Eigen::Vector2i(2 * i + 1, 2 *i));
        }

        return !finished;
    });

    bool should_close = false;
    while (!should_close) {
        should_close = !visualizer.PollEvents();
    }
    visualizer.DestroyVisualizerWindow();

    return std::make_tuple(true, result_odo);
}

std::tuple<bool, Eigen::Matrix4d, Eigen::Matrix6d>
DecoupledComputeRGBDOdometry(const RGBDImage &source, const RGBDImage &target,
                    const PinholeCameraIntrinsic &pinhole_camera_intrinsic
    /*= PinholeCameraIntrinsic()*/,
                    const Eigen::Matrix4d &odo_init /*= Eigen::Matrix4d::Identity()*/,
                    const RGBDOdometryJacobian &jacobian_method
    /*=RGBDOdometryJacobianFromHybridTerm*/,
                    const OdometryOption &option /*= OdometryOption()*/)
{
    if (!CheckRGBDImagePair(source, target)) {
        PrintError("[RGBDOdometry] Two RGBD pairs should be same in size.\n");
        return std::make_tuple(false,
                               Eigen::Matrix4d::Identity(), Eigen::Matrix6d::Zero());
    }

    std::shared_ptr<RGBDImage> source_processed, target_processed;
    std::tie(source_processed, target_processed) =
        InitializeRGBDOdometry(source, target, pinhole_camera_intrinsic,
                               odo_init, option);

    Eigen::Matrix4d extrinsic;
    bool is_success;
    std::tie(is_success, extrinsic) = ComputeMultiscale(
        *source_processed, *target_processed,
        pinhole_camera_intrinsic, odo_init, jacobian_method, option);

    if (is_success) {
        Eigen::Matrix4d trans_output = extrinsic;
        Eigen::MatrixXd info_output = CreateInformationMatrix(extrinsic,
                                                              pinhole_camera_intrinsic, source_processed->depth_,
                                                              target_processed->depth_, option);
        return std::make_tuple(true, trans_output, info_output);
    }
    else {
        return std::make_tuple(false,
                               Eigen::Matrix4d::Identity(), Eigen::Matrix6d::Identity());
    }
}

std::shared_ptr<RGBDImage> ReadRGBDImage(
    const char *color_filename, const char *depth_filename,
    const PinholeCameraIntrinsic &intrinsic,
    bool visualize) {
    Image color, depth;
    ReadImage(color_filename, color);
    ReadImage(depth_filename, depth);
    PrintDebug("Reading RGBD image : \n");
    PrintDebug("     Color : %d x %d x %d (%d bits per channel)\n",
               color.width_, color.height_,
               color.num_of_channels_, color.bytes_per_channel_ * 8);
    PrintDebug("     Depth : %d x %d x %d (%d bits per channel)\n",
               depth.width_, depth.height_,
               depth.num_of_channels_, depth.bytes_per_channel_ * 8);
    double depth_scale = 1000.0, depth_trunc = 4.0;
    bool convert_rgb_to_intensity = true;
    std::shared_ptr<RGBDImage> rgbd_image =
        CreateRGBDImageFromColorAndDepth(color,
                                         depth,
                                         depth_scale,
                                         depth_trunc,
                                         convert_rgb_to_intensity);
    if (visualize) {
        auto pcd = CreatePointCloudFromRGBDImage(*rgbd_image, intrinsic);
        DrawGeometries({pcd});
    }
    return rgbd_image;
}

int TestNativeRGBDOdometry(
    std::string source_color_path,
    std::string source_depth_path,
    std::string target_color_path,
    std::string target_depth_path) {

    using namespace open3d;
    PinholeCameraIntrinsic intrinsic = PinholeCameraIntrinsic(
        PinholeCameraIntrinsicParameters::PrimeSenseDefault);
    bool visualize = false;

    auto source = ReadRGBDImage(source_color_path.c_str(),
                                source_depth_path.c_str(),
                                intrinsic, visualize);
    auto target = ReadRGBDImage(target_color_path.c_str(),
                                target_depth_path.c_str(),
                                intrinsic, visualize);

    Eigen::Matrix4d odo_init = Eigen::Matrix4d::Identity();

    std::tuple<bool, Eigen::Matrix4d, Eigen::Matrix6d> rgbd_odo =
        DecoupledComputeRGBDOdometry(*source, *target, intrinsic, odo_init,
                            RGBDOdometryJacobianFromHybridTerm(),
                            OdometryOption({60, 60, 60}, 0.03, 0.0, 3.0));
    std::cout << "RGBD Odometry" << std::endl;
    std::cout << std::get<1>(rgbd_odo) << std::endl;
}

int main(int argc, char **argv) {
    SetVerbosityLevel(VerbosityLevel::VerboseDebug);
    std::string base_path = "/home/wei/Work/data/stanford/lounge/";
    TestNativeRGBDOdometry(base_path + "color/000004.png",
                           base_path + "depth/000004.png",
                           base_path + "color/000001.png",
                           base_path + "depth/000001.png");
//    TestNativeRGBDOdometry(base_path + "color/000366.png",
//                           base_path + "depth/000366.png",
//                           base_path + "color/000365.png",
//                           base_path + "depth/000365.png");

}