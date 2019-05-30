////
//// Created by Wei Dong on 2019-05-29.
////
//
//#include "VisualizerDR.h"
//
//namespace open3d {
//namespace visualization {
//
//bool VisualizerDR::AddGeometry(
//        std::shared_ptr<geometry::Geometry> geometry_ptr) {
//    if (geometry_ptr->GetGeometryType() ==
//        geometry::Geometry::GeometryType::ExtendedTriangleMesh) {
//        auto renderer_ptr = std::make_shared<glsl::DifferentiableRenderer>();
//        if (!(renderer_ptr->AddMutableGeometry(geometry_ptr)) {
//            utility::PrintDebug("Failed to add geometry\n");
//            return false;
//        }
//        geometry_renderer_ptrs_.emplace(renderer_ptr);
//    }
//
//    geometry_ptrs_.emplace(geometry_ptr);
//
//    view_control_ptr_->FitInGeometry(*geometry_ptr);
//    ResetViewPoint();
//    utility::PrintDebug(
//            "Add geometry and update bounding box to %s\n",
//            view_control_ptr_->GetBoundingBox().GetPrintInfo().c_str());
//    return UpdateGeometry();
//}
//
//bool VisualizerDR::CaptureBuffer(const std::string &filename, int index) {
//    auto &renderer =
//            (glsl::DifferentiableRenderer &)*geometry_renderer_ptrs_.begin();
//    renderer.CaptureBuffer(filename, index);
//    return true;
//}
//
//bool VisualizerDR::SetTargetImage(const geometry::Image &target,
//                                  const camera::PinholeCameraParameters &view) {
//    auto &renderer =
//            (glsl::DifferentiableRenderer &)*geometry_renderer_ptrs_.begin();
//    renderer.RebindTexture(target);
//    view_control_ptr_->ConvertFromPinholeCameraParameters(view);
//    return true;
//}
//
//bool VisualizerDR::UpdateLighting() {
//    auto &renderer =
//            (glsl::DifferentiableRenderer &)*geometry_renderer_ptrs_.begin();
//    renderer.UpdateEnvLighting();
//    return true;
//}
//
//float VisualizerDR::CallSGD(float lambda,
//                            bool update_albedo,
//                            bool update_material,
//                            bool update_normal) {
//    auto &renderer =
//            (glsl::DifferentiableRenderer &)*geometry_renderer_ptrs_.begin();
//    return renderer.SGD(lambda, update_albedo, update_material, update_normal);
//}
//} // visualization
//} // open3d