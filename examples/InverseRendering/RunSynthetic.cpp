//
// Created by wei on 4/28/19.
//

#include <random>
#include <Open3D/Open3D.h>
#include <InverseRendering/Visualization/Visualizer/VisualizerPBR.h>
#include <InverseRendering/Geometry/TriangleMeshExtended.h>
#include <InverseRendering/IO/ClassIO/TriangleMeshExtendedIO.h>
#include <Open3D/Utility/Console.h>
#include <InverseRendering/Geometry/ImageExt.h>
#include <InverseRendering/Visualization/Utility/DrawGeometryPBR.h>

using namespace open3d;

std::pair<std::vector<std::string>, std::vector<std::string>>
LoadDataset(const std::string &path) {
    std::vector<std::string> image_names, cam_names;
    utility::filesystem::ListFilesInDirectoryWithExtension(path, "png", image_names);
    utility::filesystem::ListFilesInDirectoryWithExtension(path, "json", cam_names);

    std::sort(image_names.begin(), image_names.end());
    std::sort(cam_names.begin(), cam_names.end());

    return std::make_pair(image_names, cam_names);
}

int main(int argc, char **argv) {
    auto result = LoadDataset("/media/wei/Data/data/pbr/image/gold_alex");
    auto image_names = result.first;
    auto cam_names = result.second;

    auto mesh = std::make_shared<geometry::TriangleMeshExtended>();
    io::ReadTriangleMeshExtendedFromPLY(
        "/media/wei/Data/data/pbr/model/sphere_gold.ply", *mesh);
//    for (auto &color : mesh->vertex_colors_) {
//        color = Eigen::Vector3d(1, 0, 0);
//    }
//
//    for (auto &material : mesh->vertex_materials_) {
//        material(0) = 1.0;
//        material(1) = 0.0;
//    }

    for (auto &normal : mesh->vertex_normals_) {
        normal = normal + Eigen::Vector3d(0.1, -0.1, 0.2);
        normal.normalize();
    }

    /** Place holder **/
    std::vector<geometry::Image> textures;
    textures.emplace_back(
        *geometry::FlipImageExt(*io::CreateImageFromFile(image_names[0])));

    auto ibl = std::make_shared<geometry::IBLLighting>();
    ibl->ReadEnvFromHDR(
        "/media/wei/Data/data/pbr/env/Alexs_Apt_2k.hdr");

    visualization::VisualizerDR visualizer;
    if (!visualizer.CreateVisualizerWindow("DR", 640, 480, 0, 0)) {
        utility::PrintWarning("Failed creating OpenGL window.\n");
        return 0;
    }
    visualizer.BuildUtilities();
    visualizer.UpdateWindowTitle();

    visualizer.AddGeometryPBR(mesh, textures, ibl);

    camera::PinholeCameraParameters cam_params;

    std::string kBasePath = "/media/wei/Data/results/";

    std::random_device rd;
    std::uniform_real_distribution<float> dist(-1, 1);
    std::uniform_real_distribution<float> dist0(0, 1);

    float lambda = 0.001;
    float loss = 0;
    for (int i = 0; i < 1; ++i) {
        for (int j = 0; j < image_names.size(); ++j) {
            auto target = geometry::FlipImageExt(*io::CreateImageFromFile(image_names[j]));
            io::ReadIJsonConvertibleFromJSON(cam_names[j], cam_params);
            visualizer.SetTargetImage(*target, cam_params);

            visualizer.UpdateRender();
            visualizer.PollEvents();

//            std::string filename = kBasePath + "iter-" + std::to_string(i) + "-img-" + std::to_string(j);
//            io::WriteImage(filename + "-origin.png", *target);
//            visualizer.CaptureBuffer(filename + "-render.png", 0);
//            visualizer.CaptureBuffer(filename + "-residual.png", 1);
//            visualizer.CaptureBuffer(filename + "-target.png", 5);
            loss += visualizer.CallSGD(lambda, true, false, false);
        }

        utility::PrintInfo("Iter %d: lambda = %f -> loss = %f\n",
                           i, lambda, loss);

        if (i % 20 == 19) {
            lambda *= 0.5f;
            io::WriteTriangleMeshExtendedToPLY(
                "mesh-iter-" + std::to_string(i) + ".ply", *mesh);
        }
    }

    float curr_loss = 0;
    float delta_c = 0.001f, delta_m;
    auto wc = Eigen::Vector3d(dist0(rd), dist0(rd), dist0(rd)).normalized();
    Eigen::Vector3d wm;
    for (int i = 0; i < 100; ++i) {
        if (dist0(rd) > 0.0005) {
            wm = ibl->SampleAround(wc, 0.02);
        } else {
            wm = Eigen::Vector3d(dist0(rd), dist0(rd), dist0(rd)).normalized();
        }
        std::cout << "wm = " << wm.transpose() << "\n";

        auto prev_value = ibl->GetValueAt(wm);
        ibl->SetValueAt(wm, prev_value + Eigen::Vector3f(dist(rd), dist(rd), dist(rd)));

        visualizer.UpdateLighting();
        curr_loss = 0;
        for (int j = 0; j < image_names.size(); ++j) {
            auto target = geometry::FlipImageExt(*io::CreateImageFromFile(image_names[j]));
            io::ReadIJsonConvertibleFromJSON(cam_names[j], cam_params);
            visualizer.SetTargetImage(*target, cam_params);

            visualizer.UpdateRender();
            visualizer.PollEvents();
            curr_loss += visualizer.CallSGD(lambda, false, false, false);
        }
        utility::PrintInfo("Iter %d: lambda = %f -> loss = %f\n",
                           i, lambda, curr_loss);

        delta_m = curr_loss - loss;
        if (delta_m > 0) {
            // backtrace
            ibl->SetValueAt(wm, prev_value);
        } else {
            utility::PrintInfo("Accept wm\n");
        }

        if ((delta_c < 0 && delta_m > delta_c) || dist0(rd) < std::abs(delta_m / delta_c)) {
            utility::PrintInfo("Update wc to wm\n");
            delta_c = delta_m;
            wc = wm;
        }

    }

    visualization::DrawGeometriesPBR({mesh}, {textures}, {ibl});
}
