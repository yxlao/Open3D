//
// Created by wei on 4/12/19.
//

#pragma once

#include <Open3D/Open3D.h>
#include "GeometryRendererPBR.h"

namespace open3d {
namespace visualization {
namespace glsl {

class DifferentiableRenderer : public GeometryRendererPBR {
public:
    ~DifferentiableRenderer() override = default;

public:
    /** Dummy **/
    bool AddGeometry(std::shared_ptr<const geometry::Geometry> geometry_ptr)
    override {};

    /** In use **/
    bool Render(const RenderOption &option, const ViewControl &view) override;
    bool AddMutableGeometry(std::shared_ptr<geometry::Geometry> &geometry_ptr);
    bool UpdateGeometry() override;

    bool CaptureBuffer(const std::string &filename, int index);

    float SGD(float lambda,
             bool update_color,
             bool update_material,
             bool update_normal);
    bool RebindTexture(const geometry::Image &image);
    bool RebindGeometry(const RenderOption &option,
                        bool rebind_color,
                        bool rebind_material,
                        bool rebind_normal);

protected:
    /** Preprocess illumination **/
    IBLVertexMapShader ibx_vertex_map_shader_;
    DifferentialShader differential_shader_;
    IndexShader index_shader_;

    BackgroundShader background_shader_;

private:
    std::shared_ptr<geometry::Geometry> mutable_geometry_ptr_;
};

} // glsl
} // visualization
} // open3d

