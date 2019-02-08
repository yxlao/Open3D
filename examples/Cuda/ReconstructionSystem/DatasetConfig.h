//
// Created by wei on 2/7/19.
//

#pragma once

#include <json/json.h>
#include <IO/IO.h>
#include <iomanip>
#include <sstream>

namespace open3d {
class DatasetConfig : public IJsonConvertible {
public:
    std::string path_dataset_;
    std::string path_intrinsic_;

    int n_frames_per_fragment_;
    int n_keyframes_per_n_frame_;

    double min_depth_;
    double max_depth_;
    double depth_factor_;
    double voxel_size_;

    double max_depth_diff_;
    double preference_loop_closure_odometry_;
    double preference_loop_closure_registration_;
    double tsdf_cubic_size_;


    PinholeCameraIntrinsic intrinsic_;
    std::vector<std::string> color_files_;
    std::vector<std::string> depth_files_;
    std::vector<std::string> fragment_files_;

    explicit DatasetConfig(
        const std::string &path_dataset = "",
        const std::string &path_intrinsic = "",
        int n_frames_per_fragment = 100,
        int n_keyframes_per_n_frame = 5,

        double min_depth = 0.3,
        double max_depth = 3.0,
        double depth_factor = 1000.0,
        double voxel_size = 0.05,

        double max_depth_diff = 0.07,
        double preference_loop_closure_odometry = 0.1,
        double preference_loop_closure_registration = 5.0,
        double tsdf_cubic_size = 3.0) :

        path_dataset_(path_dataset),
        path_intrinsic_(path_intrinsic),
        n_frames_per_fragment_(n_frames_per_fragment),
        n_keyframes_per_n_frame_(n_keyframes_per_n_frame),
        min_depth_(min_depth),
        max_depth_(max_depth),
        depth_factor_(depth_factor),
        voxel_size_(voxel_size),
        max_depth_diff_(max_depth_diff),
        preference_loop_closure_odometry_(preference_loop_closure_odometry),
        preference_loop_closure_registration_(
            preference_loop_closure_registration),
        tsdf_cubic_size_(tsdf_cubic_size) {

        if (path_intrinsic_.empty()) {
            intrinsic_ = PinholeCameraIntrinsic(
                PinholeCameraIntrinsicParameters::PrimeSenseDefault);
        } else {
            bool is_success = ReadIJsonConvertible(path_intrinsic_, intrinsic_);
            if (!is_success) {
                PrintError("Unable to read camera intrinsics: %s!\n",
                           path_intrinsic_.c_str());
            }
        }
    }

    std::vector<std::string>& GetColorFiles() {
        std::string color_directory = path_dataset_ + "/color";
        if (! filesystem::DirectoryExists(color_directory)) {
            color_directory = path_dataset_ + "/image";
            if (! filesystem::DirectoryExists(color_directory)) {
                PrintError("No color image folder found in directory %s\n",
                    color_directory.c_str());
            }
        }
        filesystem::ListFilesInDirectory(color_directory, color_files_);

        /* alphabetical order */
        std::sort(color_files_.begin(), color_files_.end());
        return color_files_;
    }

    std::vector<std::string>& GetDepthFiles() {
        std::string depth_directory = path_dataset_ + "/depth";
        if (! filesystem::DirectoryExists(depth_directory)) {
            PrintError("No depth image folder found in directory %s\n",
                       depth_directory.c_str());
        }
        filesystem::ListFilesInDirectory(depth_directory, depth_files_);

        /* alphabetical order */
        std::sort(depth_files_.begin(), depth_files_.end());
        return depth_files_;
    }

    std::vector<std::string>& GetFragmentFiles() {
        std::string fragment_directory = path_dataset_ + "/fragments_cuda";
        filesystem::ListFilesInDirectoryWithExtension(
            fragment_directory, "ply", fragment_files_);

        /* alphabetical order */
        std::sort(fragment_files_.begin(), fragment_files_.end());
        return fragment_files_;
    }

    std::shared_ptr<PoseGraph> GetPoseGraphForFragment(
        int fragment_id, bool optimized) {

        std::stringstream ss;
        ss << path_dataset_ << "/fragments_cuda/fragment_";
        if (optimized) {
            ss << "optimized_";
        }
        ss << std::setw(3) << std::setfill('0') << fragment_id;
        ss << ".json";

        return CreatePoseGraphFromFile(ss.str());
    }

    std::shared_ptr<PoseGraph> GetPoseGraphForScene(bool optimized) {
        std::stringstream ss;
        ss << path_dataset_ << "/scene_cuda/global_registration";
        if (optimized) {
            ss << "_optimized";
        }
        ss << ".json";

        return CreatePoseGraphFromFile(ss.str());
    }

    std::shared_ptr<PoseGraph> GetPoseGraphForRefinedScene(bool optimized) {
        std::stringstream ss;
        ss << path_dataset_ << "/scene_cuda/global_registration_refined";
        if (optimized) {
            ss << "_optimized";
        }
        ss << ".json";

        return CreatePoseGraphFromFile(ss.str());
    }

    std::shared_ptr<TriangleMesh> GetReconstructedScene() {
        std::stringstream ss;
        ss << path_dataset_ << "/scene_cuda/integrated.ply";

        return CreateMeshFromFile(ss.str());
    }

    bool ConvertToJsonValue(Json::Value &value) const override {}
    bool ConvertFromJsonValue(const Json::Value &value) override {
        if (!value.isObject()) {
            PrintWarning("DatasetConfig read JSON failed: unsupported json "
                         "format.\n");
            return false;
        }

        path_dataset_ = value.get("path_dataset", "").asString();
        path_intrinsic_ = value.get("path_intrinsic", "").asString();

        n_frames_per_fragment_ = value.get(
            "n_frames_per_fragment", 100).asInt();
        n_keyframes_per_n_frame_ = value.get(
            "n_keyframes_per_n_frame", 5).asInt();

        min_depth_ = value.get("min_depth", 0.3).asDouble();
        max_depth_ = value.get("max_depth", 3.0).asDouble();
        depth_factor_ = value.get("depth_factor", 1000.0).asDouble();
        voxel_size_ = value.get("voxel_size", 0.05).asDouble();

        max_depth_diff_ = value.get("max_depth_diff", 0.07).asDouble();
        preference_loop_closure_odometry_ = value.get(
            "preference_loop_closure_odometry", 0.1).asDouble();
        preference_loop_closure_registration_ = value.get(
            "preference_loop_closure_registration", 5.0).asDouble();
        tsdf_cubic_size_ = value.get("tsdf_cubic_size", 3.0).asDouble();

        return true;
    }
};
}