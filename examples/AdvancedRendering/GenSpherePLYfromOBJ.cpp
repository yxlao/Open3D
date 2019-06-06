//
// Created by wei on 4/22/19.
//

#include <AdvancedRendering/Geometry/Lighting.h>
#include <AdvancedRendering/Geometry/ExtendedTriangleMesh.h>
#include <AdvancedRendering/IO/ClassIO/ExtendedTriangleMeshIO.h>
#include <AdvancedRendering/Visualization/Utility/DrawGeometryPBR.h>
#include <Open3D/Open3D.h>

#include <Eigen/Eigen>
#include <AdvancedRendering/Geometry/TexturedTriangleMesh.h>
#include <AdvancedRendering/IO/ClassIO/TexturedTriangleMeshIO.h>

#include "data_path.h"

using namespace open3d;

Eigen::Vector3d InterpolateVec3(const geometry::Image &im, float u, float v) {
    float y = v * (im.height_ - 1);
    float x = u * (im.width_ - 1);

    int x0 = std::floor(x), x1 = std::ceil(x);
    int y0 = std::floor(y), y1 = std::ceil(y);
    float rx = x - x0, ry = y - y0;

    if (im.bytes_per_channel_ == 1) {
        Eigen::Vector3d color;
        for (int c = 0; c < 3; ++c) { // ignore the alpha channel
            color(c) =
                    *geometry::PointerAt<unsigned char>(im, y0, x0, c) * (1 - rx) * (1 - ry) +
                    *geometry::PointerAt<unsigned char>(im, y0, x1, c) * (1 - rx) * ry +
                    *geometry::PointerAt<unsigned char>(im, y1, x0, c) * rx * (1 - ry) +
                    *geometry::PointerAt<unsigned char>(im, y1, x1, c) * rx * ry;
            color(c) /= 255.0;
            color(c) = std::min(std::max(0.0, color(c)), 1.0);
        }
        return color;
    } else {
        utility::PrintError("Invalid format (%d %d)!\n", im.bytes_per_channel_, im.num_of_channels_);
        return Eigen::Vector3d::Zero();
    }
}

double InterpolateScalar(const geometry::Image &im, float u, float v) {
    float y = v * (im.height_ - 1);
    float x = u * (im.width_ - 1);

    int x0 = std::floor(x), x1 = std::ceil(x);
    int y0 = std::floor(y), y1 = std::ceil(y);
    float rx = x - x0, ry = y - y0;

    if (im.bytes_per_channel_ == 1) {
        double query =
                *geometry::PointerAt<unsigned char>(im, y0, x0, 0) * (1 - rx) * (1 - ry) +
                *geometry::PointerAt<unsigned char>(im, y0, x1, 0) * (1 - rx) * ry +
                *geometry::PointerAt<unsigned char>(im, y1, x0, 0) * rx * (1 - ry) +
                *geometry::PointerAt<unsigned char>(im, y1, x1, 0) * rx * ry;
        query /= 255.0;
        return std::min(std::max(0.0, query), 1.0);
    } else if (im.bytes_per_channel_ == 2) {
        double query = *geometry::PointerAt<unsigned short>(im, y0, x0, 0) * (1 - rx) * (1 - ry) +
                       *geometry::PointerAt<unsigned short>(im, y0, x1, 0) * (1 - rx) * ry +
                       *geometry::PointerAt<unsigned short>(im, y1, x0, 0) * rx * (1 - ry) +
                       *geometry::PointerAt<unsigned short>(im, y1, x1, 0) * rx * ry;
        query /= 65535.0;
        return std::min(std::max(0.0, query), 1.0);
    } else {
        utility::PrintError("Invalid format (%d %d)!\n", im.bytes_per_channel_, im.num_of_channels_);
        return 0;
    }
}

Eigen::Vector3d GetPositionOnSphere(float u, float v) {
    return Eigen::Vector3d(std::cos(u * 2.0f * M_PI) * std::sin(v * M_PI),
                           std::cos(v * M_PI),
                           std::sin(u * 2.0f * M_PI) * std::sin(v * M_PI));
}

int main() {
    std::string base_path = kGLTestBasePath + "/planet";

    std::string material = "gold";
    std::string prefix = base_path + "/" + material;
    std::vector<geometry::Image> textures;
    textures.push_back(*io::CreateImageFromFile(prefix + "/albedo.png"));
    textures.push_back(*io::CreateImageFromFile(prefix + "/normal.png"));
    textures.push_back(*io::CreateImageFromFile(prefix + "/metallic.png"));
    textures.push_back(*io::CreateImageFromFile(prefix + "/roughness.png"));
    textures.push_back(*io::CreateImageFromFile(prefix + "/ao.png"));

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
    ibl->ReadEnvFromHDR(kHDRPath);

    utility::SetVerbosityLevel(utility::VerbosityLevel::VerboseDebug);
    visualization::DrawGeometriesPBR({extended_mesh}, {ibl});

    io::WriteExtendedTriangleMeshToPLY(
            base_path + "/planet_" + material + ".ply", *extended_mesh);
}
