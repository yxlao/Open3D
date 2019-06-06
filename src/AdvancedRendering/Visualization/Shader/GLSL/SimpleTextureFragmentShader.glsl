#version 330 core

layout(location = 0) out vec4 color;
uniform sampler2D texture_vis;

in vec2 uv;

void main(){
    color = texture(texture_vis, uv);
}