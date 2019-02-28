//
// Created by wei on 2/21/19.
//

#pragma once

#include <vector>
#include <cmath>
#include <Core/Core.h>

std::tuple<double, double> ComputeStatistics(const std::vector<double> &vals) {
    double mean = 0;
    for (auto &val : vals) {
        mean += val;
    }
    mean /= double(vals.size());

    double std = 0;
    for (auto &val : vals) {
        std += (val - mean) * (val - mean);
    }
    std = std::sqrt(std / (vals.size() - 1));

    return std::make_tuple(mean, std);
}