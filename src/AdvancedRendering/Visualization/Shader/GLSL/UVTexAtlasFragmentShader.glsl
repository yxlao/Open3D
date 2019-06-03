#version 330 core

out vec4 FragColor;

in vec2 uv;
in vec2 frag_coord;

// material parameters
uniform sampler2D tex_image;

void main() {
    vec3 albedo = texture(tex_image, frag_coord).rgb;
    FragColor = vec4(albedo, 1.0);
}
