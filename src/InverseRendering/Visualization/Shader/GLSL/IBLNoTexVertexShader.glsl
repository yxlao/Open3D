#version 330 core

layout (location = 0) in vec3 vertex_position;
layout (location = 1) in vec3 vertex_normal;
layout (location = 2) in vec3 vertex_albedo;
layout (location = 3) in vec3 vertex_material;

out vec3 position;
out vec3 normal;
out vec3 albedo;
out vec3 tangent;
out float roughness;
out float metallic;
out float ao;

uniform mat4 P;
uniform mat4 V;
uniform mat4 M;

void main() {
    position = vec3(vec4(vertex_position, 1.0));
    normal = vertex_normal;

    albedo = vertex_albedo;

    roughness = vertex_material.r;
    metallic = vertex_material.g;
    ao = vertex_material.b;

    gl_Position =  P * V * vec4(position, 1.0);
}