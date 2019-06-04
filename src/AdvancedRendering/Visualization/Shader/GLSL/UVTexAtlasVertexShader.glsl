#version 330 core

layout(location = 0) in vec3 vertex_position;
layout(location = 1) in vec2 vertex_uv;
layout(location = 2) in vec3 vertex_normal;

out vec2 atlas_uv;     /* for writing atlas value */
out vec4 ref_position; /* for depth test */

out vec3 position;
out vec3 normal;

uniform mat4 P;
uniform mat4 V;
uniform mat4 M;

void main() {
    atlas_uv = vertex_uv;
    gl_Position =  vec4(2 * atlas_uv - 1, 0, 1);

    ref_position = P * V * M * vec4(vertex_position, 1.0);

    normal = (V * M * vec4(vertex_normal, 0)).xyz;
    position = (V * M * vec4(vertex_position, 1.0)).xyz;
}