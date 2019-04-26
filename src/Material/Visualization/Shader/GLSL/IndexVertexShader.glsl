#version 330 core

in vec3 vertex_position;
flat out int index;

uniform mat4 V;
uniform mat4 P;

void main() {
    gl_Position =  P * V * vec4(vertex_position, 1.0);
    index = gl_VertexID;
}
