////
//// Created by wei on 2/4/19.
////
//
//#include <vector>
//#include <string>
//
//#include <Open3D/Open3D.h>
//#include <Cuda/Open3DCuda.h>
//
//#include "DatasetConfig.h"
//#include <AdvancedRendering/Geometry/TriangleMeshExtended.h>
//#include <AdvancedRendering/Visualization/Visualizer/VisualizerPBR.h>
//#include <AdvancedRendering/Visualization/Utility/DrawGeometryPBR.h>
//
//using namespace open3d;
//using namespace open3d::utility;
//using namespace open3d::io;
//using namespace open3d::registration;
//
//namespace RenderScene {
//void RenderSceneInFragment(
//    visualization::VisualizerPBR &visualizer,
//    int fragment_id,
//    DatasetConfig &config) {
//
//    PoseGraph global_pose_graph;
//    ReadPoseGraph(config.GetPoseGraphFileForRefinedScene(true),
//                  global_pose_graph);
//
//    PoseGraph local_pose_graph;
//    ReadPoseGraph(config.GetPoseGraphFileForFragment(fragment_id, true),
//                  local_pose_graph);
//
//    cuda::PinholeCameraIntrinsicCuda intrinsics(config.intrinsic_);
//    const int begin = fragment_id * config.n_frames_per_fragment_;
//    const int end = std::min((fragment_id + 1) * config.n_frames_per_fragment_,
//                             (int) config.color_files_.size());
//
//    camera::PinholeCameraParameters params;
//    params.intrinsic_ = config.intrinsic_;
//    for (int i = begin; i < end; ++i) {
//        PrintDebug("Rendering frame %d ...\n", i);
//
//        /* Use ground truth trajectory */
//        auto pose = global_pose_graph.nodes_[fragment_id].pose_
//            * local_pose_graph.nodes_[i - begin].pose_;
//        params.extrinsic_ = pose.inverse();
//
//        visualizer.GetViewControl().ConvertFromPinholeCameraParameters(params);
//        visualizer.UpdateRender();
//        visualizer.PollEvents();
//
//        auto rendered_fbos = visualizer.geometry_renderer_fbo_outputs_[0];
//        auto index_map = rendered_fbos[0];
//    }
//}
//
//int Run(DatasetConfig &config) {
//    auto mesh = std::make_shared<geometry::TriangleMesh>();
//    io::ReadTriangleMeshFromPLY(config.GetPlyFileForFragment(0), *mesh);
//
//    auto mesh_extended = std::make_shared<geometry::TriangleMeshExtended>();
//    mesh_extended->vertices_ = mesh->vertices_;
//    mesh_extended->vertex_colors_ = mesh->vertex_colors_;
//    mesh_extended->vertex_normals_ = mesh->vertex_normals_;
//    mesh_extended->triangles_ = mesh->triangles_;
//    mesh_extended->vertex_materials_.resize(mesh->vertices_.size());
//    for (auto &mat : mesh_extended->vertex_materials_) {
//        mat = Eigen::Vector3d(1, 0, 1);
//    }
//
//    std::vector<geometry::Image> textures; /** dummy **/
//    textures.emplace_back(*geometry::FlipImageExt(*io::ReadImage(config.color_files_[0])));
//
//    auto ibl = std::make_shared<geometry::IBLLighting>();
//    ibl->ReadEnvFromHDR("/media/wei/Data/data/pbr/env/White.hdr");
//
//    bool is_success = config.GetFragmentFiles();
//    if (!is_success) {
//        utility::PrintError("Unable to get fragment files\n");
//        return -1;
//    }
//
//    visualization::VisualizerDR visualizer;
//    if (!visualizer.CreateVisualizerWindow("PBR", 640, 480, 0, 0)) {
//        PrintWarning("Failed creating OpenGL window.\n");
//        return 0;
//    }
//    visualizer.BuildUtilities();
//    visualizer.UpdateWindowTitle();
//
//    visualizer.AddGeometryPBR(mesh_extended, textures, ibl);
//    visualizer.Run();
//
//    for (int i = 0; i < config.fragment_files_.size(); ++i) {
//        RenderSceneInFragment(visualizer, i, config);
//    }
//
//    return 0;
//}
//}