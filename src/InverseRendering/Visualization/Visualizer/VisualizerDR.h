////
//// Created by Wei Dong on 2019-05-29.
////
//
//#pragma once
//
//#include <VisualizerPBR.h>
//
//namespace open3d {
//namespace visualization {
//class VisualizerDR : public VisualizerPBR {
//public:
//    virtual bool AddGeometry(
//        std::shared_ptr<geometry::Geometry> geometry_ptr);
//
//    bool CaptureBuffer(const std::string &filename, int index);
//
//    bool SetTargetImage(const geometry::Image &target,
//                        const camera::PinholeCameraParameters &view);
//    bool UpdateLighting();
//    float CallSGD(float lambda,
//                  bool update_albedo,
//                  bool update_material,
//                  bool update_normal);
//};
//}
//}