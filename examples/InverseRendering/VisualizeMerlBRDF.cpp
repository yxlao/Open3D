//
// Created by wei on 4/10/19.
//

#include <Open3D/Open3D.h>
#include <InverseRendering/Geometry/MerlBRDF.h>

using namespace open3d;

int main(int argc, char **argv) {
    MerlBRDF brdf;
    brdf.ReadFromBinary("/media/wei/Data/data/merl/pink-fabric.binary");

    auto pcl = std::make_shared<geometry::PointCloud>();

    double gamma = 2.2;
    double theta_in = M_PI / 2.2;
    double phi_in = 0;
    Eigen::Vector3d in = Eigen::Vector3d(std::cos(theta_in) * std::cos(phi_in),
                                         std::cos(theta_in) * std::sin(phi_in),
                                         std::sin(theta_in));
    Eigen::Vector3d out = Eigen::Vector3d(std::cos(theta_in) * std::cos(M_PI + phi_in),
                                         std::cos(theta_in) * std::sin(M_PI + phi_in),
                                         std::sin(theta_in));
    int n = 128;
    for (int k = 0; k < n; k++) {
        double theta_out = k * 0.5 * M_PI / n;
        for (int l = 0; l < 4 * n; l++) {
            double phi_out = l * 2.0 * M_PI / (4 * n);
            Eigen::Vector3d color = brdf.Query(
                theta_in, phi_in, theta_out, phi_out);
            color = Eigen::Vector3d(
                std::pow(color(0), 1.0 / gamma),
                std::pow(color(1), 1.0 / gamma),
                std::pow(color(2), 1.0 / gamma));
            Eigen::Vector3d position = Eigen::Vector3d(
                std::cos(theta_out) * std::cos(phi_out),
                std::cos(theta_out) * std::sin(phi_out),
                std::sin(theta_out));
            pcl->points_.emplace_back(position);
            pcl->colors_.emplace_back(color);
        }
    }

    auto line = std::make_shared<geometry::LineSet>();
    line->points_.push_back(Eigen::Vector3d::Zero());
    line->points_.push_back(2 * in);
    line->points_.push_back(2 * out);

    line->lines_.push_back(Eigen::Vector2i(0, 1));
    line->colors_.push_back(Eigen::Vector3d(0, 0, 1));

    line->lines_.push_back(Eigen::Vector2i(0, 2));
    line->colors_.push_back(Eigen::Vector3d(1, 0, 0));
    visualization::DrawGeometries({pcl, line});

}