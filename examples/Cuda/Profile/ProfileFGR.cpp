//
// Created by wei on 2/21/19.
//

#include <vector>
#include <string>
#include <Core/Core.h>
#include <IO/IO.h>

#include <Cuda/Registration/RegistrationCuda.h>
#include <Cuda/Registration/ColoredICPCuda.h>
#include <Cuda/Registration/FastGlobalRegistrationCuda.h>

#include <Core/Registration/PoseGraph.h>
#include <Core/Registration/GlobalOptimization.h>
#include <Core/Registration/FastGlobalRegistration.h>

#include "examples/Cuda/DatasetConfig.h"
#include "Analyzer.h"

using namespace open3d;

void ProfileFGR(DatasetConfig &config, bool use_cuda) {
    std::vector<double> fgr_times;

    int num_fragments = config.thumbnail_fragment_files_.size();
    for (int s = 0; s < num_fragments; ++s) {
        auto source =
            CreatePointCloudFromFile(config.thumbnail_fragment_files_[s]);

        for (int t = s + 2; t < num_fragments; ++t) {
            auto target = CreatePointCloudFromFile(
                config.thumbnail_fragment_files_[t]);

            Match match;
            match.s = s;
            match.t = t;

            Timer fgr_timer;

            if (use_cuda) {
                fgr_timer.Start();
                cuda::FastGlobalRegistrationCuda fgr;
                fgr.Initialize(*source, *target);

                auto result = fgr.ComputeRegistration();
                match.trans_source_to_target = result.transformation_;

                /**!!! THIS SHOULD BE REFACTORED !!!**/
                cuda::RegistrationCuda registration(
                    TransformationEstimationType::PointToPoint);
                auto source_copy = *source;
                source_copy.Transform(result.transformation_);
                registration.Initialize(source_copy, *target,
                                        config.voxel_size_ * 1.4f);
                registration.transform_source_to_target_ =
                    result.transformation_;
                match.information = registration.ComputeInformationMatrix();
                fgr_timer.Stop();
            } else {
                fgr_timer.Start();
                auto source_fpfh = ComputeFPFHFeature(
                    *source, open3d::KDTreeSearchParamHybrid(0.25, 100));
                auto target_fpfh = ComputeFPFHFeature(
                    *target, open3d::KDTreeSearchParamHybrid(0.25, 100));
                auto result = FastGlobalRegistration(*source, *target,
                                       *source_fpfh, *target_fpfh);
                fgr_timer.Stop();
            }

            double time = fgr_timer.GetDuration();
            fgr_times.push_back(time);

            PrintInfo("Fragment %d - %d takes %f ms\n", s, t, time);
        }
    }

    double mean, std;
    std::tie(mean, std) = ComputeStatistics(fgr_times);
    PrintInfo("total time: avg = %f, std = %f\n", mean, std);
}


int main(int argc, char **argv) {
    DatasetConfig config;

    std::string config_path = argc > 1 ? argv[1] :
                              "/home/wei/Work/projects/dense_mapping/Open3D/examples/Cuda"
                              "/ReconstructionSystem/config/fr2_desktop.json";

    bool is_success = ReadIJsonConvertible(config_path, config);
    if (!is_success) return 1;
    is_success = config.GetThumbnailFragmentFiles();

    ProfileFGR(config, true);
    ProfileFGR(config, false);
}