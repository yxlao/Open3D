//
// Created by wei on 11/14/18.
//

#include <string>
#include <vector>
#include <sstream>
#include <Core/Core.h>
#include <IO/IO.h>
#include <Cuda/Odometry/SequentialRGBDOdometryCuda.h>
#include <Cuda/Integration/ScalableTSDFVolumeCuda.h>
#include <Cuda/Integration/ScalableMeshVolumeCuda.h>
#include <Cuda/Geometry/PointCloudCuda.h>
#include <Visualization/Visualization.h>

#include <opencv2/opencv.hpp>
#include <thread>

void WriteLossesToLog(std::string filename,
                      std::vector<std::vector<float>> &losses) {
    FILE *f = fopen(filename.c_str(), "w");
    if (f == NULL) {
        open3d::PrintError("Unable to open file %s, aborted.\n",
                           filename.c_str());
        return;
    }

    for (auto &loss : losses) {
        for (auto &l : loss) {
            fprintf(f, "%f ", l);
        }
        fprintf(f, "\n");
    }
    fclose(f);
}

int main(int argc, char **argv) {
    using namespace open3d;

    /** Load data **/
    std::string base_path = "/home/wei/Work/data/stanford/lounge/";
    Image source_color, source_depth, target_color, target_depth;

    /** Prepare odometry **/
    RGBDOdometryCuda<3> odometry;
    odometry.SetIntrinsics(PinholeCameraIntrinsic(
        PinholeCameraIntrinsicParameters::PrimeSenseDefault));
    odometry.SetParameters(0.986f, 0.01f, 3.0f, 0.03f);

    for (int step = 1; step < 2; ++step) {
        PrintInfo("Step: %d\n", step);
        for (int i = 1; i + step < 3000; ++i) {
            PrintInfo("%d\n", i);
            std::stringstream ss;

            ss.str("");
            ss << base_path << "color/"
               << std::setw(6) << std::setfill('0') << i << ".png";
            ReadImage(ss.str(), target_color);

            ss.str("");
            ss << base_path << "depth/"
               << std::setw(6) << std::setfill('0') << i << ".png";
            ReadImage(ss.str(), target_depth);

            ss.str("");
            ss << base_path << "color/"
               << std::setw(6) << std::setfill('0') << i + step << ".png";
            ReadImage(ss.str(), source_color);

            ss.str("");
            ss << base_path << "depth/"
               << std::setw(6) << std::setfill('0') << i + step << ".png";
            ReadImage(ss.str(), source_depth);

            RGBDImageCuda source, target;
            source.Upload(source_depth, source_color);
            target.Upload(target_depth, target_color);
            odometry.PrepareData(source, target);
            odometry.transform_source_to_target_ = Eigen::Matrix4d::Identity();

            std::vector<std::vector<float>> losses;
            losses.resize(3);
            for (int level = 2; level >= 0; --level) {
                for (int iter = 0; iter < 100; ++iter) {
                    float loss = odometry.ApplyOneIterationOnLevel(level, iter);
                    losses[level].push_back(loss);
                }
            }

            ss.str("");
            ss << "odometry-" << i << "-step-" << step << ".log";
            WriteLossesToLog(ss.str(), losses);
        }
    }

    return 0;
}