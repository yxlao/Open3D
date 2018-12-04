//
// Created by wei on 12/3/18.
//

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
#include "ReadDataAssociation.h"

#include <opencv2/opencv.hpp>
#include <thread>

using namespace open3d;

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
    double depth_scale = 5000.0, depth_trunc = 4.0;
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

}

int main(int argc, char **argv) {
    SetVerbosityLevel(VerbosityLevel::VerboseDebug);
    std::string base_path = "/home/wei/Work/data/tum/rgbd_dataset_freiburg3_long_office_household";

    auto rgbd_filenames = ReadDataAssociation(
        base_path + "/data_association.txt");

    PinholeCameraTrajectory trajectory_gt;

    /** This API loads camera_to_world and turns to world_to_camera
     * (with inverse)**/
    ReadPinholeCameraTrajectoryFromLOG(
        base_path + "/trajectory.log", trajectory_gt);

    for (auto &param : trajectory_gt.parameters_) {
        param.extrinsic_ = param.extrinsic_.inverse();
    }

    /** This API directly saves camera_to_world (without inverse) **/
    WritePinholeCameraTrajectoryToLOG("trajectory_gt.log", trajectory_gt);

    PinholeCameraIntrinsic intrinsic = PinholeCameraIntrinsic(
        640, 480, 535.4, 539.2, 320.1, 247.6);

    Eigen::Matrix4d target_to_world = trajectory_gt.parameters_[0].extrinsic_;

    PinholeCameraTrajectory trajectory;

    int nframes = rgbd_filenames.size();
    for (int i = 0; i < nframes; ++i) {
        bool visualize = false;

        if (i >= 1) {
            auto source = ReadRGBDImage(
                (base_path + "/" + rgbd_filenames[i].second).c_str(),
                (base_path + "/" + rgbd_filenames[i].first).c_str(),
                intrinsic, visualize);
            auto target = ReadRGBDImage(
                (base_path + "/" + rgbd_filenames[i - 1].second).c_str(),
                (base_path + "/" + rgbd_filenames[i - 1].first).c_str(),
                intrinsic, visualize);

            Eigen::Matrix4d odo_init = Eigen::Matrix4d::Identity();

            std::tuple<bool, Eigen::Matrix4d, Eigen::Matrix6d> rgbd_odo =
                ComputeRGBDOdometry(*source, *target, intrinsic, odo_init,
                                    RGBDOdometryJacobianFromHybridTerm(),
                                    OdometryOption({20, 10, 5}, 0.07, 0.01, 4.0)
                                    );

            Eigen::Matrix4d source_to_target = std::get<1>(rgbd_odo);
            target_to_world = target_to_world * source_to_target;
        }

        PinholeCameraParameters params;
        params.intrinsic_ = PinholeCameraIntrinsic(
            PinholeCameraIntrinsicParameters::PrimeSenseDefault);
        params.extrinsic_ = target_to_world;
        trajectory.parameters_.emplace_back(params);
    }
    WritePinholeCameraTrajectoryToLOG("trajectory_cpp.log", trajectory);
}