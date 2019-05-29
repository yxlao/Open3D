//
// Created by wei on 4/15/19.
//

#pragma once

#include <Open3D/Open3D.h>
#include <InverseRendering/Geometry/Lighting.h>
#include <InverseRendering/Visualization/Shader/LightingPreprocessRenderer.h>
#include "RenderOptionWithLighting.h"

namespace open3d {
namespace visualization {

class VisualizerPBR : public VisualizerWithKeyCallback {
public:
    /** Handle geometry (including textures) **/
    virtual bool AddGeometry(
        std::shared_ptr<const geometry::Geometry> geometry_ptr) override;

    /** Handling lighting (shared over the visualizer) **/
    virtual bool InitRenderOption() override {
        render_option_ptr_ = std::unique_ptr<RenderOptionWithLighting>(
            new RenderOptionWithLighting);
        return true;
    }

    /** Call this function
     * - AFTER @CreateVisualizerWindow (where @InitRenderOption is called)
     * - BEFORE @Run (or whatever customized rendering task) **/
    bool BuildLighting(const std::shared_ptr<geometry::Lighting> &lighting) {
        auto &render_option_ptr_with_lighting =
            (std::unique_ptr<RenderOptionWithLighting> &)
                render_option_ptr_;
        render_option_ptr_with_lighting->lighting_ptr_ = lighting;

        if (lighting->GetLightingType()
            != geometry::Lighting::LightingType::IBL) {
            return true;
        }

        if (light_preprocessing_renderer_ptr_ == nullptr) {
            light_preprocessing_renderer_ptr_ =
                std::make_shared<glsl::LightingPreprocessRenderer>();
        }

        bool success = light_preprocessing_renderer_ptr_->RenderToOption(
            *render_option_ptr_, *view_control_ptr_);
        return true;
    }

    /** This renderer:
     * 1. Preprocess input HDR lighting image,
     * 2. Maintain textures,
     *    (a buffer that should be propagate to option instantly)
     * 3. Destroy context on leave. **/
    std::shared_ptr<glsl::LightingPreprocessRenderer>
        light_preprocessing_renderer_ptr_ = nullptr;
};

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
}  // namespace visualization
}  // namespace open3d
