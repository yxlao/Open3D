//
// Created by wei on 4/15/19.
//

#include <Open3D/Open3D.h>
#include <Material/Physics/TriangleMeshPhysics.h>
#include <Material/Physics/Lighting.h>
#include <Material/Visualization/Utility/DrawGeometryPBR.h>

using namespace open3d;
int main() {
    std::string base_path = "/media/wei/Data/data/pbr/textures/pbr/rusted_iron";

    auto mesh = std::make_shared<geometry::TriangleMeshPhysics>();
    const unsigned int X_SEGMENTS = 16;
    const unsigned int Y_SEGMENTS = 16;
    const float PI = 3.14159265359;
    for (unsigned int y = 0; y <= Y_SEGMENTS; ++y) {
        for (unsigned int x = 0; x <= X_SEGMENTS; ++x) {
            float xSegment = (float) x / (float) X_SEGMENTS;
            float ySegment = (float) y / (float) Y_SEGMENTS;
            float xPos = std::cos(xSegment * 2.0f * PI) * std::sin(ySegment * PI);
            float yPos = std::cos(ySegment * PI);
            float zPos = std::sin(xSegment * 2.0f * PI) * std::sin(ySegment * PI);

            mesh->vertices_.emplace_back(Eigen::Vector3d(xPos, yPos, zPos));
            mesh->vertex_normals_.emplace_back(Eigen::Vector3d(xPos, yPos, zPos));
            mesh->vertex_uvs_.emplace_back(Eigen::Vector2f(xSegment, ySegment));
        }
    }

    std::vector<unsigned int> indices;
    bool oddRow = false;
    for (int y = 0; y < Y_SEGMENTS; ++y) {
        if (!oddRow) // even rows: y == 0, y == 2; and so on
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
        mesh->triangles_.emplace_back(Eigen::Vector3i(indices[i + 1], indices[i], indices[i + 2]));
        mesh->triangles_.emplace_back(Eigen::Vector3i(indices[i + 1], indices[i + 2], indices[i + 3]));
    }

    std::vector<geometry::Image> textures;
    textures.push_back(*io::CreateImageFromFile(base_path + "/albedo.png"));
    textures.push_back(*io::CreateImageFromFile(base_path + "/normal.png"));
    textures.push_back(*io::CreateImageFromFile(base_path + "/metallic.png"));
    textures.push_back(*io::CreateImageFromFile(base_path + "/roughness.png"));
    textures.push_back(*io::CreateImageFromFile(base_path + "/ao.png"));

    auto lighting = std::make_shared<physics::SpotLighting>();
    lighting->light_positions_ = {Eigen::Vector3f(2, 1, 0), Eigen::Vector3f(1, 2, 0)};
    lighting->light_colors_ = {Eigen::Vector3f(1, 1, 1), Eigen::Vector3f(1, 1, 1)};

    utility::SetVerbosityLevel(utility::VerbosityLevel::VerboseDebug);
    visualization::DrawGeometriesPBR({mesh}, {textures}, {lighting});

    io::WriteTriangleMeshToPLY(base_path + "/sphere.ply", *mesh, true);
}