//
// Created by wei on 4/22/19.
//

#include <AdvancedRendering/Geometry/Lighting.h>
#include <AdvancedRendering/Geometry/ExtendedTriangleMesh.h>
#include <AdvancedRendering/IO/ClassIO/ExtendedTriangleMeshIO.h>
#include <AdvancedRendering/Visualization/Utility/DrawGeometryPBR.h>
#include <Open3D/Open3D.h>

#include <Eigen/Eigen>

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
        for (int c = 0; c < 3; ++c) {
            color(c) =
                    *geometry::PointerAt<unsigned char>(im, y0, x0, c) * (1 - rx) * (1 - ry) +
                    *geometry::PointerAt<unsigned char>(im, y0, x1, c) * (1 - rx) * ry +
                    *geometry::PointerAt<unsigned char>(im, y1, x0, c) * rx * (1 - ry) +
                    *geometry::PointerAt<unsigned char>(im, y1, x1, c) * rx * ry;
            color(c) /= 255.0;
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
        return query / 255.0;
    } else if (im.bytes_per_channel_ == 2) {
        auto query = *geometry::PointerAt<unsigned short>(im, y0, x0, 0) * (1 - rx) * (1 - ry) +
                     *geometry::PointerAt<unsigned short>(im, y0, x1, 0) * (1 - rx) * ry +
                     *geometry::PointerAt<unsigned short>(im, y1, x0, 0) * rx * (1 - ry) +
                     *geometry::PointerAt<unsigned short>(im, y1, x1, 0) * rx * ry;
        return query / 65535.0;
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
    auto mesh = std::make_shared<geometry::ExtendedTriangleMesh>();

    // path on mac
    //    std::string base_path = "/Users/dongw1/Work/Data/planet";

    // path on workstation
    std::string base_path = kGLTestBasePath + "/planet";

    std::string material = "plastic";
    std::string prefix = base_path + "/" + material;
    std::vector<geometry::Image> textures;
    textures.push_back(*io::CreateImageFromFile(prefix + "/albedo.png"));
    textures.push_back(*io::CreateImageFromFile(prefix + "/normal.png"));
    textures.push_back(*io::CreateImageFromFile(prefix + "/metallic.png"));
    textures.push_back(*io::CreateImageFromFile(prefix + "/roughness.png"));
    textures.push_back(*io::CreateImageFromFile(prefix + "/ao.png"));

    /** This should be higher to store more texture information **/
    const unsigned int X_SEGMENTS = 16;
    const unsigned int Y_SEGMENTS = 16;

    const float du = 1.0f / 1000000;
    for (unsigned int vi = 0; vi <= Y_SEGMENTS; ++vi) {
        for (unsigned int ui = 0; ui <= X_SEGMENTS; ++ui) {
            float u = (float) ui / (float) X_SEGMENTS;
            float v = (float) vi / (float) Y_SEGMENTS;

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

            mesh->vertex_textures_.emplace_back(
                    Eigen::Vector3d(roughness, metallic, ao));

            mesh->vertex_uvs_.emplace_back(
                    Eigen::Vector2d(u, v));
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
    ibl->ReadEnvFromHDR(kHDRPath);

    utility::SetVerbosityLevel(utility::VerbosityLevel::VerboseDebug);
    visualization::DrawGeometriesPBR({mesh}, {ibl});

//    io::WriteExtendedTriangleMeshToPLY(base_path + "/sphere.ply", *mesh);

    return 0;
}
