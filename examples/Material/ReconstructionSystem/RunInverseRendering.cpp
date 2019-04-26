//
// Created by wei on 2/4/19.
//

#include "DatasetConfig.h"

#include "RenderScene.h"

using namespace open3d;
using namespace open3d::io;
using namespace open3d::utility;

std::string SecondsToHMS(double seconds) {
    int minutes = int(seconds / 60);
    int hours = minutes / 60;

    double seconds_remain = seconds - minutes * 60;

    std::stringstream ss;
    ss << std::setw(2) << std::setfill('0') << hours << ":"
       << std::setw(2) << std::setfill('0') << (minutes % 60) << ":";

    ss.setf(std::ios::fixed);
    ss << std::setw(6) << std::setfill('0')
       << std::setprecision(3) << seconds_remain;

    return ss.str();
}

int main(int argc, char **argv) {
    DatasetConfig config;

    std::string config_path = argc > 1 ? argv[1] :
        kDefaultDatasetConfigDir + "/stanford/lounge.json";

    bool is_success = ReadIJsonConvertible(config_path, config);
    if (!is_success) return 1;

    Timer timer_total;
    timer_total.Start();

    Timer timer;

    timer.Start();
    RenderScene::Run(config);
    timer.Stop();
    std::string render_scene_time = SecondsToHMS(timer.GetDuration() * 1e-3);

    timer_total.Stop();
    std::string total_time = SecondsToHMS(timer_total.GetDuration() * 1e-3);

    PrintInfo("================================\n");
    PrintInfo(" - Render scene       : %s\n", render_scene_time.c_str());
    PrintInfo(" - Total              : %s\n", total_time.c_str());
    PrintInfo("================================\n");

    return 0;
}