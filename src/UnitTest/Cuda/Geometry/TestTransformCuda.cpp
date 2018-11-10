//
// Created by wei on 10/3/18.
//

#include <Cuda/Common/TransformCuda.h>
#include <Eigen/Eigen>
#include <Core/Core.h>

#include "UnitTest.h"

TEST(TransformCuda, Transform) {
    using namespace open3d;

    for (int i = 0; i < 1000; ++i) {
        /* Generate random R & t */
        Eigen::Vector3d w = Eigen::Vector3d::Random();
        float theta = w.norm();
        w = w / theta;
        Eigen::Matrix3d w_tilde;
        w_tilde << 0, -w(2), w(1),
            w(2), 0, -w(0),
            -w(1), w(0), 0;
        Eigen::Matrix3d R =
            Eigen::Matrix3d::Identity()
            + sin(theta) * w_tilde + (1 - cos(theta)) * (w_tilde * w_tilde);
        
        Eigen::Vector3d t = Eigen::Vector3d::Random();
        Eigen::Matrix4d T = Eigen::Matrix4d::Identity();
        T.block<3, 3>(0, 0) = R;
        T.block<3, 1>(0, 3) = t;


        TransformCuda transform_cuda;
        transform_cuda.FromEigen(R, t);
        float
            matrix_norm = (T.inverse() - transform_cuda.Inverse().ToEigen()).norm();
        EXPECT_LE(matrix_norm, 1e-6);

        Vector3f v_cuda;
        Eigen::Vector3d v = Eigen::Vector3d::Random();
        v_cuda.FromEigen(v);

        Vector3f Tv_cuda = transform_cuda * v_cuda;
        float vector_norm = ((T * v.homogeneous()).hnormalized() -
            (transform_cuda * v_cuda).ToEigen()).norm();
        EXPECT_LE(vector_norm, 1e-6);
    }
    PrintInfo("Transform tests passed\n");

    Vector1f v;
    v(0) = 1.0f;
    v = 1.0f * v;
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
