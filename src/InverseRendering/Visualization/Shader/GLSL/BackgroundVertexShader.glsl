#version 330 core

layout (location = 0) in vec3 vertex_position;

uniform mat4 P;
uniform mat4 V;

out vec3 position;

void main() {
    position = vertex_position;

	mat4 R = mat4(mat3(V));
	vec4 clipPos = P * R * vec4(position, 1.0);

	gl_Position = clipPos.xyww;
}