//
// Created by wei on 10/10/18.
//

#pragma once

//template<size_t N>
//__device__
//inline float UniformTSDFVolumeCudaServer<N>::tsdf_interp(float x, float y, float z) {
//    Vector3f Xf = world_to_volume(x, y, z);
//    if (! InVolumef(Xf)) return 0;
//
//    Vector3i X(int(floor(Xf(0))), int(floor(Xf(1))), int(floor(Xf(2))));
//    Vector3f r = Xf - X.ToVectorf();
//
//    float tsdf_weight = 0;
//    float tsdf_value = 0;
//#pragma unroll 1
//    for (int i = 0; i < 8; ++i) {
//        Vector3i dX(i & 1, (i & 2) >> 1, (i & 4) >> 2);
//        Vector3i Xi = X + dX;
//        bool in_volume = InVolume(Xi);
//        int index = IndexOf(Xi);
//        float weight = in_volume ?
//                       (dX(0) * r(0) + (1 - dX(0)) * (1 - r(0))) * /*0 -> 1 - r, 1 - > r*/
//                           (dX(1) * r(1) + (1 - dX(1)) * (1 - r(1))) *
//                           (dX(2) * r(2) + (1 - dX(2)) * (1 - r(2)))
//                                 : 0;
//
//        tsdf_weight += weight;
//        tsdf_value += in_volume ? weight * tsdf_[index] : 0;
//    }
//
//    return (tsdf_weight == 0) ? 0 : tsdf_value / tsdf_weight;
//}


