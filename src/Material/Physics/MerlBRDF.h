//
// Created by wei on 4/3/19.
//

#pragma once

#include <string>
#include <vector>
#include <Open3D/Open3D.h>

class MerlBRDF {

#define BRDF_SAMPLING_RES_THETA_H       90
#define BRDF_SAMPLING_RES_THETA_D       90
#define BRDF_SAMPLING_RES_PHI_D         360
#define RED_SCALE   (1.0/1500.0)
#define GREEN_SCALE (1.15/1500.0)
#define BLUE_SCALE  (1.66/1500.0)

public:
    int theta_h_resolution_ = -1; /* half angle elevation (azimuth fixed) */
    int theta_d_resolution_ = -1; /* diff angle elevation */
    int phi_d_resolution_ = -1; /* diff angle azimuth */

    int total_samples_ = -1;

    bool ReadFromBinary(const std::string &filename);
    Eigen::Vector3d Query(double theta_in, double phi_in,
                          double theta_out, double phi_out);

private:
    std::vector<double> data_;
};


