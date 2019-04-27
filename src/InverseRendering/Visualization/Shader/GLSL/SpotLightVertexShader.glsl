#version 330 core

layout (location = 0) in vec3 vertex_position;
layout (location = 1) in vec3 vertex_normal;
layout (location = 2) in vec2 vertex_uv;

out vec3 position;
out vec3 normal;
out vec2 uv;

uniform mat4 M;
uniform mat4 V;
uniform mat4 P;

void main() {
    uv = vertex_uv;
    position = vec3(M * vec4(vertex_position, 1.0));
    normal = mat3(M) * vertex_normal;

    gl_Position = P * V * vec4(position, 1.0);
}