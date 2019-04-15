#include "MerlBRDF.h"

// Copyright 2005 Mitsubishi Electric Research Laboratories All Rights Reserved.

// Permission to use, copy and modify this software and its documentation without
// fee for educational, research and non-profit purposes, is hereby granted, provided
// that the above copyright notice and the following three paragraphs appear in all copies.

// To request permission to incorporate this software into commercial products contact:
// Vice President of Marketing and Business Development;
// Mitsubishi Electric Research Laboratories (MERL), 201 Broadway, Cambridge, MA 02139 or
// <license@merl.com>.

// IN NO EVENT SHALL MERL BE LIABLE TO ANY PARTY FOR DIRECT, INDIRECT, SPECIAL, INCIDENTAL,
// OR CONSEQUENTIAL DAMAGES, INCLUDING LOST PROFITS, ARISING OUT OF THE USE OF THIS SOFTWARE AND
// ITS DOCUMENTATION, EVEN IF MERL HAS BEEN ADVISED OF THE POSSIBILITY OF SUCH DAMAGES.

// MERL SPECIFICALLY DISCLAIMS ANY WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
// WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE.  THE SOFTWARE PROVIDED
// HEREUNDER IS ON AN "AS IS" BASIS, AND MERL HAS NO OBLIGATIONS TO PROVIDE MAINTENANCE, SUPPORT,
// UPDATES, ENHANCEMENTS OR MODIFICATIONS.

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// cross product of two vectors
void cross_product(double *v1, double *v2, double *out) {
    out[0] = v1[1] * v2[2] - v1[2] * v2[1];
    out[1] = v1[2] * v2[0] - v1[0] * v2[2];
    out[2] = v1[0] * v2[1] - v1[1] * v2[0];
}

// normalize vector
void normalize(double *v) {
    // normalize
    double len = sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
    v[0] = v[0] / len;
    v[1] = v[1] / len;
    v[2] = v[2] / len;
}

// rotate vector along one axis
void rotate_vector(double *vector, double *axis, double angle, double *out) {
    double temp;
    double cross[3];
    double cos_ang = cos(angle);
    double sin_ang = sin(angle);

    out[0] = vector[0] * cos_ang;
    out[1] = vector[1] * cos_ang;
    out[2] = vector[2] * cos_ang;

    temp = axis[0] * vector[0] + axis[1] * vector[1] + axis[2] * vector[2];
    temp = temp * (1.0 - cos_ang);

    out[0] += axis[0] * temp;
    out[1] += axis[1] * temp;
    out[2] += axis[2] * temp;

    cross_product(axis, vector, cross);

    out[0] += cross[0] * sin_ang;
    out[1] += cross[1] * sin_ang;
    out[2] += cross[2] * sin_ang;
}

// convert standard coordinates to half vector/difference vector coordinates
void std_coords_to_half_diff_coords(double theta_in,
                                    double fi_in,
                                    double theta_out,
                                    double fi_out,
                                    double &theta_half,
                                    double &fi_half,
                                    double &theta_diff,
                                    double &fi_diff) {

    // compute in vector
    double in_vec_z = cos(theta_in);
    double proj_in_vec = sin(theta_in);
    double in_vec_x = proj_in_vec * cos(fi_in);
    double in_vec_y = proj_in_vec * sin(fi_in);
    double in[3] = {in_vec_x, in_vec_y, in_vec_z};
    normalize(in);


    // compute out vector
    double out_vec_z = cos(theta_out);
    double proj_out_vec = sin(theta_out);
    double out_vec_x = proj_out_vec * cos(fi_out);
    double out_vec_y = proj_out_vec * sin(fi_out);
    double out[3] = {out_vec_x, out_vec_y, out_vec_z};
    normalize(out);


    // compute halfway vector
    double half_x = (in_vec_x + out_vec_x) / 2.0f;
    double half_y = (in_vec_y + out_vec_y) / 2.0f;
    double half_z = (in_vec_z + out_vec_z) / 2.0f;
    double half[3] = {half_x, half_y, half_z};
    normalize(half);

    // compute  theta_half, fi_half
    theta_half = acos(half[2]);
    fi_half = atan2(half[1], half[0]);

    double bi_normal[3] = {0.0, 1.0, 0.0};
    double normal[3] = {0.0, 0.0, 1.0};
    double temp[3];
    double diff[3];

    // compute diff vector
    rotate_vector(in, normal, -fi_half, temp);
    rotate_vector(temp, bi_normal, -theta_half, diff);

    // compute  theta_diff, fi_diff
    theta_diff = acos(diff[2]);
    fi_diff = atan2(diff[1], diff[0]);

}

// Lookup theta_half index
// This is a non-linear mapping!
// In:  [0 .. pi/2]
// Out: [0 .. 89]
inline int theta_half_index(double theta_half) {
    if (theta_half <= 0.0)
        return 0;
    double theta_half_deg =
        ((theta_half / (M_PI / 2.0)) * BRDF_SAMPLING_RES_THETA_H);
    double temp = theta_half_deg * BRDF_SAMPLING_RES_THETA_H;
    temp = sqrt(temp);
    int ret_val = (int) temp;
    if (ret_val < 0) ret_val = 0;
    if (ret_val >= BRDF_SAMPLING_RES_THETA_H)
        ret_val = BRDF_SAMPLING_RES_THETA_H - 1;
    return ret_val;
}

// Lookup theta_diff index
// In:  [0 .. pi/2]
// Out: [0 .. 89]
inline int theta_diff_index(double theta_diff) {
    int tmp = int(theta_diff / (M_PI * 0.5) * BRDF_SAMPLING_RES_THETA_D);
    if (tmp < 0)
        return 0;
    else if (tmp < BRDF_SAMPLING_RES_THETA_D - 1)
        return tmp;
    else
        return BRDF_SAMPLING_RES_THETA_D - 1;
}

// Lookup phi_diff index
inline int phi_diff_index(double phi_diff) {
    // Because of reciprocity, the BRDF is unchanged under
    // phi_diff -> phi_diff + M_PI
    if (phi_diff < 0.0)
        phi_diff += M_PI;

    // In: phi_diff in [0 .. pi]
    // Out: tmp in [0 .. 179]
    int tmp = int(phi_diff / M_PI * BRDF_SAMPLING_RES_PHI_D / 2);
    if (tmp < 0)
        return 0;
    else if (tmp < BRDF_SAMPLING_RES_PHI_D / 2 - 1)
        return tmp;
    else
        return BRDF_SAMPLING_RES_PHI_D / 2 - 1;
}

// Given a pair of incoming/outgoing angles, look up the BRDF.
Eigen::Vector3d MerlBRDF::Query(double theta_in,
                                double phi_in,
                                double theta_out,
                                double phi_out) {
    Eigen::Vector3d value = Eigen::Vector3d(0);

    // Convert to halfangle / difference angle coordinates
    double theta_half, phi_half, theta_diff, phi_diff;

    std_coords_to_half_diff_coords(
        theta_in, phi_in, theta_out, phi_out,
        theta_half, phi_half, theta_diff, phi_diff);

    // Find index.
    // Note that phi_half is ignored, since isotropic BRDFs are assumed
    int ind = phi_diff_index(phi_diff) +
        theta_diff_index(theta_diff) * phi_d_resolution_ +
        theta_half_index(theta_half) * phi_d_resolution_ * theta_d_resolution_;

    value(0) = data_[ind] * RED_SCALE;
    value(1) = data_[ind + total_samples_] * GREEN_SCALE;
    value(2) = data_[ind + total_samples_ * 2] * BLUE_SCALE;

    if (value(0) < 0.0 || value(1) < 0.0 || value(2) < 0.0)
        open3d::utility::PrintError("Below horizon.\n");

    return value;
}

// Read BRDF data
bool MerlBRDF::ReadFromBinary(const std::string &filename) {
    FILE *fid = fopen(filename.c_str(), "rb");
    if (!fid) return false;

    fread(&theta_h_resolution_, sizeof(int), 1, fid); /* 90 */
    fread(&theta_d_resolution_, sizeof(int), 1, fid); /* 90 */
    fread(&phi_d_resolution_, sizeof(int), 1, fid);   /* 180 */

    total_samples_ = theta_h_resolution_
        * theta_d_resolution_ * phi_d_resolution_;
    if (theta_h_resolution_ != BRDF_SAMPLING_RES_THETA_H
        || theta_d_resolution_ != BRDF_SAMPLING_RES_THETA_D
        || phi_d_resolution_ != BRDF_SAMPLING_RES_PHI_D / 2) {
        open3d::utility::PrintError("Dimensions don't match\n");
        fclose(fid);
        return false;
    }

    data_.resize(3 * total_samples_);
    fread(data_.data(), sizeof(double), 3 * total_samples_, fid);

    fclose(fid);
    return true;
}

//int main(int argc, char *argv[]) {
//    const char *filename = argv[1];
//    double *brdf;
//
//    // read brdf
//    if (!read_brdf(filename, brdf)) {
//        fprintf(stderr, "Error reading %s\n", filename);
//        exit(1);
//    }
//
//    // print out a 16x64x16x64 table table of BRDF values
//    const int n = 16;
//    for (int i = 0; i < n; i++) {
//        double theta_in = i * 0.5 * M_PI / n;
//        for (int j = 0; j < 4 * n; j++) {
//            double phi_in = j * 2.0 * M_PI / (4 * n);
//            for (int k = 0; k < n; k++) {
//                double theta_out = k * 0.5 * M_PI / n;
//                for (int l = 0; l < 4 * n; l++) {
//                    double phi_out = l * 2.0 * M_PI / (4 * n);
//                    double red, green, blue;
//                    lookup_brdf_val(brdf,
//                                    theta_in,
//                                    phi_in,
//                                    theta_out,
//                                    phi_out,
//                                    red,
//                                    green,
//                                    blue);
//                    printf("%f %f %f\n",
//                           (float) red,
//                           (float) green,
//                           (float) blue);
//                }
//            }
//        }
//    }
//    return 0;
//}
//
