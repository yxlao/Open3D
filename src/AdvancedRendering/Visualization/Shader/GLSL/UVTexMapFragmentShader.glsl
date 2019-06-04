#version 330 core

layout(location = 0) out vec3 FragColor;

in vec2 uv;
in vec3 position;

// material parameters
uniform sampler2D tex_albedo;

void main() {
    vec3 albedo = texture(tex_albedo, uv).rgb;
    FragColor = vec3(albedo);
}
