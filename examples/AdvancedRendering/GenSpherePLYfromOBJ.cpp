//
// Created by wei on 4/22/19.
//

#include <AdvancedRendering/Geometry/Lighting.h>
#include <AdvancedRendering/Geometry/ExtendedTriangleMesh.h>
#include <AdvancedRendering/IO/ClassIO/ExtendedTriangleMeshIO.h>
#include <AdvancedRendering/Visualization/Utility/DrawGeometryPBR.h>
#include <Open3D/Open3D.h>

#include <Eigen/Eigen>
#include <opencv2/opencv.hpp>
#include <AdvancedRendering/Geometry/TexturedTriangleMesh.h>
#include <AdvancedRendering/IO/ClassIO/TexturedTriangleMeshIO.h>

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
    std::string base_path = "/Users/dongw1/Work/Data/planet";
    std::string material = "plastic";
    std::string prefix = base_path + "/" + material + "/";
    std::vector<cv::Mat> textures;
    textures.push_back(
        cv::imread(prefix + "albedo.png", cv::IMREAD_UNCHANGED));
    textures.push_back(
        cv::imread(prefix + "normal.png", cv::IMREAD_UNCHANGED));
    textures.push_back(
        cv::imread(prefix + "metallic.png", cv::IMREAD_UNCHANGED));
    textures.push_back(
        cv::imread(prefix + "roughness.png", cv::IMREAD_UNCHANGED));
    textures.push_back(
        cv::imread(prefix + "ao.png", cv::IMREAD_UNCHANGED));

    auto mesh = std::make_shared<geometry::TexturedTriangleMesh>();
    io::ReadTexturedTriangleMeshFromOBJ(base_path + "/planet.obj", *mesh);
    auto extended_mesh = std::make_shared<geometry::ExtendedTriangleMesh>();

    extended_mesh->vertices_ = mesh->vertices_;
    extended_mesh->vertex_normals_ = mesh->vertex_normals_;
    extended_mesh->vertex_uvs_ = mesh->vertex_uvs_;
    extended_mesh->triangles_ = mesh->triangles_;

    extended_mesh->vertex_colors_.resize(mesh->vertex_uvs_.size());
    extended_mesh->vertex_textures_.resize(mesh->vertex_uvs_.size());

    double du = 0.0001;
    for (int i = 0; i < mesh->vertex_uvs_.size(); ++i) {
        Eigen::Vector2d uv = mesh->vertex_uvs_[i];
        double u = uv[0], v = uv[1];

        Eigen::Vector3d N = mesh->vertex_normals_[i];
        Eigen::Vector3d T = GetPositionOnSphere(u + du, v) -
            GetPositionOnSphere(u - du, v);
        Eigen::Vector3d B = N.cross(T);
        N.normalize();
        T.normalize();
        B.normalize();

        Eigen::Vector3d tangent_normal = InterpolateVec3(textures[1], u, v);
        tangent_normal = 2 * tangent_normal - Eigen::Vector3d::Ones();
        N = Eigen::Vector3d(tangent_normal[0] * T
                                + tangent_normal[1] * B
                                + tangent_normal[2] * N);
        extended_mesh->vertex_normals_[i] = N;

        auto albedo = InterpolateVec3(textures[0], u, v);
        extended_mesh->vertex_colors_[i] = albedo;

        auto metallic = InterpolateScalar(textures[2], u, v);
        auto roughness = InterpolateScalar(textures[3], u, v);
        auto ao = InterpolateScalar(textures[4], u, v);

        extended_mesh->vertex_textures_[i] = Eigen::Vector3d(roughness, metallic, ao);
    }

    auto ibl = std::make_shared<geometry::IBLLighting>();
    ibl->ReadEnvFromHDR(
        "/Users/dongw1/Work/Data/resources/textures/hdr/newport_loft.hdr");

    utility::SetVerbosityLevel(utility::VerbosityLevel::VerboseDebug);
    visualization::DrawGeometriesPBR({extended_mesh}, {ibl});

    io::WriteExtendedTriangleMeshToPLY(
        base_path + "/planet_" + material + ".ply",
        *extended_mesh);
}