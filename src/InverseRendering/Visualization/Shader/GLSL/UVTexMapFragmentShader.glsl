#version 330 core

out vec4 FragColor;

in vec2 uv;
in vec3 position;

// material parameters
uniform sampler2D tex_albedo;

const float PI = 3.14159265359;

void main() {
    vec3 albedo = texture(tex_albedo, uv).rgb;
    FragColor = vec4(albedo, 1.0);
}
