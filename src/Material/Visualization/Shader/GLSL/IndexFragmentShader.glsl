#version 330 core

layout(location = 0) out int FragColor;
flat in int index;

void main() {
    FragColor = index;
}
