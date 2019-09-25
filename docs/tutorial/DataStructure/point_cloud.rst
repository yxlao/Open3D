.. _point_cloud:

PointCloud
==========

PointCloud creation
-------------------

.. code-block:: cpp

    #include "Open3D.h"
    using namespace std;
    using namespace open3d;

    int main() {
        auto pcd = std::make_shared<geometry::PointCloud>();
        pcd->points_.push_back({0.0, 0.0, 0.0});  // (x, y, z)
        pcd->points_.push_back({1.0, 1.0, 1.0});  // (x, y, z)
        pcd->colors_.push_back({1.0, 0.0, 0.0});  // Red
        pcd->colors_.push_back({1.0, 0.0, 0.0});  // Red
        visualization::DrawGeometries({pcd});
        return 0;
    }

Set color to PointCloud
-----------------------


