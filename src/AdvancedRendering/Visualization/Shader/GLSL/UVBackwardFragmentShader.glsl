#version 330 core

layout(location = 0) out vec3 color;
layout(location = 1) out vec3 weight;

in vec2 atlas_uv;
in vec4 ref_position;

in vec3 normal;
in vec3 position;

uniform sampler2D tex_image;
uniform sampler2D tex_depthmap;

uniform float margin;
uniform float cos_thr;

void main() {
    vec3 proj_ref_position = ref_position.xyz / ref_position.w;
    proj_ref_position = proj_ref_position * 0.5 + 0.5;

    float closest_depth = texture(tex_depthmap, proj_ref_position.xy).r;
    float current_depth = proj_ref_position.z;

    float cos_np = dot(normal, -position);

    bool mask =
    /* clipping */
    (proj_ref_position.x >= 0 && proj_ref_position.x <= 1) &&
    (proj_ref_position.y >= 0 && proj_ref_position.y <= 1) &&
    /* angle truncation */
    (cos_np > cos_thr)
    /* adaptive occlusion test */
    && (current_depth - closest_depth
    < margin * (cos_np - cos_thr) / (1 - cos_thr));

    color = mask ? texture(tex_image, proj_ref_position.xy).xyz : vec3(0);
    weight = mask ? vec3(0, min(0.1 * cos_np / dot(position, position), 1), 1) : vec3(0);
}
