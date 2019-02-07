//
// Created by wei on 2/4/19.
//

#include <vector>
#include <string>

#include <Core/Core.h>
#include <IO/IO.h>

#include <Core/Registration/Registration.h>
#include <Core/Registration/PoseGraph.h>

#include <Cuda/Integration/ScalableTSDFVolumeCuda.h>
#include <Cuda/Integration/ScalableMeshVolumeCuda.h>

#include "System.h"
#include "../Utils.h"

using namespace open3d;

void IntegrateFragment(
    int fragment_id,
    const std::string &base_path,
    const std::vector<std::pair<std::string, std::string>> &filenames,
    cuda::ScalableTSDFVolumeCuda<8> &volume) {

    PoseGraph global_pose_graph;
    ReadPoseGraph(
        GetScenePoseGraphName(base_path, "_refined_optimized"),
//        GetScenePoseGraphName(base_path, "_optimized"),
        global_pose_graph);

    PoseGraph local_pose_graph;
    ReadPoseGraph(
        GetFragmentPoseGraphName(fragment_id, base_path,"optimized_"),
        local_pose_graph);

    cuda::PinholeCameraIntrinsicCuda intrinsics(
        PinholeCameraIntrinsicParameters::PrimeSenseDefault);
    cuda::RGBDImageCuda rgbd(kDepthMin, kDepthMax, kDepthFactor);

    const int begin = fragment_id * kFramesPerFragment;
    const int end = std::min((fragment_id + 1) * kFramesPerFragment,
                             (int) filenames.size());

    for (int i = begin; i < end; ++i) {
        PrintDebug("Integrating frame %d ...\n", i);

        Image depth, color;
        ReadImage(base_path + "/" + filenames[i].first, depth);
        ReadImage(base_path + "/" + filenames[i].second, color);
        rgbd.Upload(depth, color);

        /* Use ground truth trajectory */
        Eigen::Matrix4d pose = global_pose_graph.nodes_[fragment_id].pose_
            * local_pose_graph.nodes_[i - begin].pose_;
        cuda::TransformCuda trans;
        trans.FromEigen(pose);

        volume.Integrate(rgbd, intrinsics, trans);
    }
}

int main(int argc, char **argv) {
    SetVerbosityLevel(VerbosityLevel::VerboseDebug);

    Timer timer;
    timer.Start();
    auto rgbd_filenames = ReadDataAssociation(
        kBasePath + "/data_association.txt");

    cuda::TransformCuda trans = cuda::TransformCuda::Identity();
    cuda::ScalableTSDFVolumeCuda<8> tsdf_volume(
        40000, 600000, kCubicSize / 512, kTSDFTruncation, trans);

    for (int i = 0; i < kNumFragments; ++i) {
        IntegrateFragment(i, kBasePath, rgbd_filenames, tsdf_volume);
    }

    tsdf_volume.GetAllSubvolumes();
    cuda::ScalableMeshVolumeCuda<8> mesher(
        tsdf_volume.active_subvolume_entry_array().size(),
        cuda::VertexWithNormalAndColor, 20000000, 40000000);
    mesher.MarchingCubes(tsdf_volume);
    auto mesh = mesher.mesh().Download();

    WriteTriangleMeshToPLY(GetScenePlyName(kBasePath), *mesh);
    timer.Stop();
    PrintInfo("IntegrateScene takes %.3f s\n", timer.GetDuration() / 1000.0f);

    DrawGeometries({mesh});

    return 0;
}