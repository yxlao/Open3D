//
// Created by wei on 4/22/19.
//

#include <InverseRendering/Geometry/Lighting.h>
#include <InverseRendering/Geometry/TriangleMeshExtended.h>
#include <InverseRendering/IO/ClassIO/TriangleMeshExtendedIO.h>
#include <InverseRendering/Visualization/Utility/DrawGeometryPBR.h>
#include <Open3D/Open3D.h>

#include <Eigen/Eigen>
#include <opencv2/opencv.hpp>

using namespace open3d;

Eigen::Vector3d InterpolateVec3(const cv::Mat &im, float u, float v) {
    float y = v * (im.rows - 1);
    float x = u * (im.cols - 1);

    int x0 = std::floor(x), x1 = std::ceil(x);
    int y0 = std::floor(y), y1 = std::ceil(y);
    float rx = x - x0, ry = y - y0;

    if (im.type() == CV_8UC3) {
        auto query = im.at<cv::Vec3b>(y0, x0) * (1 - rx) * (1 - ry) +
                     im.at<cv::Vec3b>(y0, x1) * (1 - rx) * ry +
                     im.at<cv::Vec3b>(y1, x0) * rx * (1 - ry) +
                     im.at<cv::Vec3b>(y1, x1) * rx * ry;
        return Eigen::Vector3d(query[2], query[1], query[0]) / 255.0;
    } else if (im.type() == CV_8UC4) {
        auto query = im.at<cv::Vec4b>(y0, x0) * (1 - rx) * (1 - ry) +
                     im.at<cv::Vec4b>(y0, x1) * (1 - rx) * ry +
                     im.at<cv::Vec4b>(y1, x0) * rx * (1 - ry) +
                     im.at<cv::Vec4b>(y1, x1) * rx * ry;
        return Eigen::Vector3d(query[2], query[1], query[0]) / 255.0;
    } else {
        utility::PrintError("Invalid format (%d %d)!\n", im.depth(),
                            im.channels());
        return Eigen::Vector3d::Zero();
    }
}

double InterpolateScalar(const cv::Mat &im, float u, float v) {
    float y = v * (im.rows - 1);
    float x = u * (im.cols - 1);

    int x0 = std::floor(x), x1 = std::ceil(x);
    int y0 = std::floor(y), y1 = std::ceil(y);
    float rx = x - x0, ry = y - y0;

    if (im.type() == CV_8UC1) {
        auto query = im.at<uchar>(y0, x0) * (1 - rx) * (1 - ry) +
                     im.at<uchar>(y0, x1) * (1 - rx) * ry +
                     im.at<uchar>(y1, x0) * rx * (1 - ry) +
                     im.at<uchar>(y1, x1) * rx * ry;
        return query / 255.0;
    } else if (im.type() == CV_16UC1) {
        auto query = im.at<ushort>(y0, x0) * (1 - rx) * (1 - ry) +
                     im.at<ushort>(y0, x1) * (1 - rx) * ry +
                     im.at<ushort>(y1, x0) * rx * (1 - ry) +
                     im.at<ushort>(y1, x1) * rx * ry;
        return query / 65535.0;
    } else if (im.type() == CV_8UC3) {
        auto query = im.at<cv::Vec3b>(y0, x0) * (1 - rx) * (1 - ry) +
                     im.at<cv::Vec3b>(y0, x1) * (1 - rx) * ry +
                     im.at<cv::Vec3b>(y1, x0) * rx * (1 - ry) +
                     im.at<cv::Vec3b>(y1, x1) * rx * ry;
        return query[0] / 255.0;
    } else if (im.type() == CV_8UC4) {
        auto query = im.at<cv::Vec4b>(y0, x0) * (1 - rx) * (1 - ry) +
                     im.at<cv::Vec4b>(y0, x1) * (1 - rx) * ry +
                     im.at<cv::Vec4b>(y1, x0) * rx * (1 - ry) +
                     im.at<cv::Vec4b>(y1, x1) * rx * ry;
        return query[0] / 255.0;
    } else if (im.type() == CV_16UC3) {
        auto query = im.at<cv::Vec3s>(y0, x0) * (1 - rx) * (1 - ry) +
                     im.at<cv::Vec3s>(y0, x1) * (1 - rx) * ry +
                     im.at<cv::Vec3s>(y1, x0) * rx * (1 - ry) +
                     im.at<cv::Vec3s>(y1, x1) * rx * ry;
        return query[0] / 65535.0;
    } else {
        utility::PrintError("Invalid format (%d %d)!\n", im.depth(),
                            im.channels());
        return 0;
    }
}

Eigen::Vector3d GetPositionOnSphere(float u, float v) {
    return Eigen::Vector3d(std::cos(u * 2.0f * M_PI) * std::sin(v * M_PI),
                           std::cos(v * M_PI),
                           std::sin(u * 2.0f * M_PI) * std::sin(v * M_PI));
}

int main() {
    auto mesh = std::make_shared<geometry::TriangleMeshExtended>();

    std::string base_path =
            "/Users/dongw1/Work/Data/resources/textures/pbr/gold";
    std::vector<cv::Mat> textures;
    textures.push_back(
            cv::imread(base_path + "/albedo.png", cv::IMREAD_UNCHANGED));
    textures.push_back(
            cv::imread(base_path + "/normal.png", cv::IMREAD_UNCHANGED));
    textures.push_back(
            cv::imread(base_path + "/metallic.png", cv::IMREAD_UNCHANGED));
    textures.push_back(
            cv::imread(base_path + "/roughness.png", cv::IMREAD_UNCHANGED));
    textures.push_back(cv::imread(base_path + "/ao.png", cv::IMREAD_UNCHANGED));

    for (int i = 0; i < 5; ++i) {
        std::cout << textures[i].depth() << " " << textures[i].channels()
                  << "\n";
    }

    /** This should be higher to store more texture information **/
    const unsigned int X_SEGMENTS = 16;
    const unsigned int Y_SEGMENTS = 16;

    const float du = 1.0f / 1000000;
    for (unsigned int vi = 0; vi <= Y_SEGMENTS; ++vi) {
        for (unsigned int ui = 0; ui <= X_SEGMENTS; ++ui) {
            float u = (float)ui / (float)X_SEGMENTS;
            float v = (float)vi / (float)Y_SEGMENTS;

            Eigen::Vector3d position = GetPositionOnSphere(u, v);

            mesh->vertices_.emplace_back(position);

            Eigen::Vector3d N = position;
            Eigen::Vector3d T = GetPositionOnSphere(u + du, v) -
                                GetPositionOnSphere(u - du, v);
            Eigen::Vector3d B = N.cross(T);
            N.normalize();
            T.normalize();
            B.normalize();

            Eigen::Vector3d tangent_normal = InterpolateVec3(textures[1], u, v);
            tangent_normal = 2 * tangent_normal - Eigen::Vector3d::Ones();
            mesh->vertex_normals_.emplace_back(N);
            Eigen::Vector3d(tangent_normal[0] * T + tangent_normal[1] * B +
                            tangent_normal[2] * N);

            auto albedo = InterpolateVec3(textures[0], u, v);
            mesh->vertex_colors_.emplace_back(albedo);

            auto metallic = InterpolateScalar(textures[2], u, v);
            auto roughness = InterpolateScalar(textures[3], u, v);
            auto ao = InterpolateScalar(textures[4], u, v);

            mesh->vertex_materials_.emplace_back(
                    Eigen::Vector3d(roughness, metallic, ao));
        }
    }

    std::vector<unsigned int> indices;
    bool oddRow = false;
    for (int y = 0; y < Y_SEGMENTS; ++y) {
        if (!oddRow)  // even rows: y == 0, y == 2; and so on
        {
            for (int x = 0; x <= X_SEGMENTS; ++x) {
                indices.push_back(y * (X_SEGMENTS + 1) + x);
                indices.push_back((y + 1) * (X_SEGMENTS + 1) + x);
            }
        } else {
            for (int x = X_SEGMENTS; x >= 0; --x) {
                indices.push_back((y + 1) * (X_SEGMENTS + 1) + x);
                indices.push_back(y * (X_SEGMENTS + 1) + x);
            }
        }
        oddRow = !oddRow;
    }

    for (int i = 0; i < indices.size() - 2; i += 2) {
        mesh->triangles_.emplace_back(
                Eigen::Vector3i(indices[i + 1], indices[i], indices[i + 2]));
        mesh->triangles_.emplace_back(Eigen::Vector3i(
                indices[i + 1], indices[i + 2], indices[i + 3]));
    }

    auto ibl = std::make_shared<geometry::IBLLighting>();
    ibl->ReadEnvFromHDR(
            "/Users/dongw1/Work/Data/resources/textures/hdr/newport_loft.hdr");

    std::vector<geometry::Image> textures_dummy;
    utility::SetVerbosityLevel(utility::VerbosityLevel::VerboseDebug);
    visualization::DrawGeometriesPBR({mesh}, {textures_dummy}, {ibl});

    return 0;
}
