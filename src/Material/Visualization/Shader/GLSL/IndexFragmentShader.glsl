#version 330 core

out int FragColor;
flat in int index;

void main() {
    FragColor = index;
}
