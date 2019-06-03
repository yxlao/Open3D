#version 330 core

layout(location = 0) in vec3 vertex_position;
layout(location = 1) in vec2 vertex_uv;

out vec2 uv;
out vec2 frag_coord;

uniform mat4 P;
uniform mat4 V;
uniform mat4 M;

void main() {
    uv = vertex_uv;
    vec4 position = P * V * M * vec4(vertex_position, 1.0);
    frag_coord = 0.5 + 0.5 * position.xy / position.w;

    gl_Position =  vec4(2 * uv - 1, 0, 1);
    //    P * V * vec4(position, 1.0);
}