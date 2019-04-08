// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// The MIT License (MIT)
//
// Copyright (c) 2018 www.open3d.org
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
// FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
// IN THE SOFTWARE.
// ----------------------------------------------------------------------------

#include <iostream>
#include <memory>
#include <thread>
#include <string>

#include <Open3D/Open3D.h>
using namespace open3d;

int main() {
    std::string pcd_path =
            "/Users/yixing/repo/Open3D/examples/TestData/ICP/cloud_bin_0.pcd";
    auto pcd = std::make_shared<geometry::PointCloud>();
    io::ReadPointCloud(pcd_path, *pcd);

    visualization::VisualizerWithEditing vis;
    // auto vis = visualization::Visualizer();
    vis.CreateVisualizerWindow("Visualizer");
    vis.AddGeometry(pcd);

    // size_t step = 300 while (True) {
    // if len(np.asarray(pcd.points)) < step:
    //             break
    //         pcd.points =
    //         o3d.utility.Vector3dVector(np.asarray(pcd.points)[:-step])
    //         pcd.colors =
    //         o3d.utility.Vector3dVector(np.asarray(pcd.colors)[:-step])
    //         vis.update_geometry()
    //         vis.poll_events()
    //         vis.update_renderer()
    //         print(len(np.asarray(pcd.points)))
    // }

    return 0;
}
