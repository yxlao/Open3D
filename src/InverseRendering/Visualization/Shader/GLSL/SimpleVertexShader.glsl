#version 330 core

layout (location = 0) in vec3 vertex_position;

out vec3 position;

uniform mat4 P;
uniform mat4 V;

void main() {
    position = vertex_position;  
    gl_Position =  P * V * vec4(position, 1.0);
}