//
// Created by wei on 2/4/19.
//


void MakePoseGraphForScene(const std::vector<std::string> &ply_filenames) {
    // adjacent: colored icp, initialized with pose_graph_frag.nodes[-1].pose
    // non-adjacent: fgr, skip unreasonable solutions

    // transformation: source_to_target
    // odometry: world_to_source
    // pose_graph appends camera[source|target]_to_world

    //    if t == s + 1: # odometry case
    //        odometry = np.dot(transformation, odometry)
    //    odometry_inv = np.linalg.inv(odometry)
    //
    //    pose_graph.nodes.append(PoseGraphNode(odometry_inv))
    //    pose_graph.edges.append(
    //        PoseGraphEdge(s, t, transformation,
    //                      information, uncertain = False))
    //    else: # loop closure case
    //        pose_graph.edges.append(
    //            PoseGraphEdge(s, t, transformation,
    //                          information, uncertain = True))
    //        return (odometry, pose_graph)
}