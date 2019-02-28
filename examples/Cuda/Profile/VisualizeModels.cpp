//
// Created by wei on 2/22/19.
//

#include <Core/Core.h>
#include <Visualization/Visualization.h>
#include <IO/IO.h>

using namespace open3d;

std::string path = "/media/wei/TOSHIBA EXT/IROS-2019/bundlefusion/";

//"/home/wei/";

int main(int argc, char **argv) {
    VisualizerWithKeyCallback visualizer;
    if (!visualizer.CreateVisualizerWindow("RGBDOdometry", 1280, 960, 0, 0)) {
        PrintWarning("Failed creating OpenGL window.\n");
        return -1;
    }
    visualizer.BuildUtilities();
    visualizer.UpdateWindowTitle();

    std::vector<std::shared_ptr<TriangleMesh>> meshes;
    meshes.resize(2);
    meshes[0] = CreateMeshFromFile(path + "office3_ours.ply");
//    meshes[1] = CreateMeshFromFile(path + "copyroom_ours.ply");
    meshes[1] = CreateMeshFromFile(path + "office3_bf.ply");

    std::shared_ptr<TriangleMesh> mesh_ptr = std::make_shared<TriangleMesh>();
    *mesh_ptr = *meshes[0];
    visualizer.AddGeometry(mesh_ptr);

    int idx = 0;
    visualizer.RegisterKeyCallback(GLFW_KEY_SPACE, [&](Visualizer *vis) {
        idx = (idx + 1) % 2;
        printf("%d\n", idx);
        *mesh_ptr = *meshes[idx];
        vis->UpdateGeometry();
        return true;
    });

    bool should_close = false;
    while (!should_close) {
        should_close = !visualizer.PollEvents();
    }
    visualizer.DestroyVisualizerWindow();

    return 0;
}