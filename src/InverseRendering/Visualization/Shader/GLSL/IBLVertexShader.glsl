#version 330 core
in vec3 vertex_position;
in vec2 vertex_uv;
in vec3 vertex_normal;

out vec2 uv;
out vec3 position;
out vec3 normal;

uniform mat4 P;
uniform mat4 V;
uniform mat4 M;

void main() {
    uv = vertex_uv;
    position = vec3(M * vec4(vertex_position, 1.0));
    normal = mat3(M) * vertex_normal;

    gl_Position =  P * V * vec4(position, 1.0);
}