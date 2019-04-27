#version 330 core

layout(location = 0) out vec4 FragColor;
layout(location = 1) out vec3 dRoughness;
layout(location = 2) out vec3 dAlbedo;
layout(location = 3) out vec3 dNx;
layout(location = 4) out vec3 dNy;
layout(location = 5) out vec3 dNz;


in vec3 position;
in vec3 normal;
in vec3 albedo;
in float roughness;
in float metallic;
in float ao;

// IBL
uniform samplerCube tex_env_diffuse;
uniform samplerCube tex_env_specular;
uniform sampler2D   tex_lut_specular;

uniform vec3 camera_position;

const float PI = 3.14159265359;

// ----------------------------------------------------------------------------
float GeometrySchlickGGX(float NdotV, float roughness) {
    float r = (roughness + 1.0);
    float k = (r*r) / 8.0;

    float nom   = NdotV;
    float denom = NdotV * (1.0 - k) + k;

    return nom / denom;
}
// ----------------------------------------------------------------------------
float GeometrySmith(vec3 N, vec3 V, vec3 L, float roughness) {
    float NdotV = max(dot(N, V), 0.0);
    float NdotL = max(dot(N, L), 0.0);
    float ggx2 = GeometrySchlickGGX(NdotV, roughness);
    float ggx1 = GeometrySchlickGGX(NdotL, roughness);

    return ggx1 * ggx2;
}

// ----------------------------------------------------------------------------
vec3 FresnelSchlickRoughness(float cosTheta, vec3 F0, float roughness) {
    return F0 + (max(vec3(1.0 - roughness), F0) - F0) * pow(1.0 - cosTheta, 5.0);
}

// ----------------------------------------------------------------------------
vec3 Diffuse(vec3 albedo_degamma, vec3 N) {
    vec3 preconv_diffuse = texture(tex_env_diffuse, N).rgb;
    return preconv_diffuse * albedo_degamma;
}

vec3 Specular(float NoV, vec3 R, vec3 F, float roughness) {
    const float MAX_REFLECTION_LOD = 4.0;
    vec3 prefiltered_color = textureLod(tex_env_specular, R, roughness * MAX_REFLECTION_LOD).rgb;
    vec2 env_brdf  = texture(tex_lut_specular, vec2(NoV, roughness)).rg;
    return prefiltered_color * (F * env_brdf.x + env_brdf.y);
}

vec3 Color(vec3 V, vec3 albedo, float roughness, vec3 N) {
    vec3 albedo_degamma = pow(albedo, vec3(2.2));
    vec3 R = reflect(-V, N);
    float NoV = max(dot(N, V), 0.0);

    // Diffuse
    vec3 diffuse = Diffuse(albedo_degamma, N);

    // Diffuse/Specular Ratio; Fresnel
    vec3 F0 = mix(vec3(0.04), albedo_degamma, metallic);
    vec3 F = FresnelSchlickRoughness(NoV, F0, roughness);
    vec3 kS = F;
    vec3 kD = 1.0 - kS;
    kD *= 1.0 - metallic;

    // Specular
    vec3 specular = Specular(NoV, R, F, roughness);

    // Result
    vec3 color = (kD * diffuse + specular) * ao;
    color = pow(color, vec3(1.0/2.2));

    return color;
}

// ----------------------------------------------------------------------------
void main() {
    // Viewpoints
    vec3 V = normalize(camera_position - position);
    vec3 N = normal;

    // Output rendering
    vec3 color = Color(V, albedo, roughness, N);
    FragColor = vec4(color, 1.0);

    // roughness difference
    const float delta = 0.001f;
    dRoughness = Color(V, albedo, roughness + delta, N)
               - Color(V, albedo, roughness - delta, N);
    dRoughness /= (2 * delta);

    // albedo difference, channels spearated
    vec3 dR = Color(V, albedo + vec3(delta, 0, 0), roughness, N)
            - Color(V, albedo - vec3(delta, 0, 0), roughness, N);
    vec3 dG = Color(V, albedo + vec3(0, delta, 0), roughness, N)
            - Color(V, albedo - vec3(0, delta, 0), roughness, N);
    vec3 dB = Color(V, albedo + vec3(0, 0, delta), roughness, N)
            - Color(V, albedo - vec3(0, 0, delta), roughness, N);
    dAlbedo = vec3(dR.r, dG.g, dB.b) / (2 * delta);

    // so3 difference
    // [ 1   -dz   dy]
    // [ dz   1   -dx]
    // [-dy   dx   1 ]
    vec3 dex = delta * vec3(0, -N.z, N.y);
    vec3 dey = delta * vec3(N.z, 0, -N.x);
    vec3 dez = delta * vec3(-N.y, N.x, 0);
    vec3 dNx = Color(V, albedo, roughness, N + dex)
             - Color(V, albedo, roughness, N - dex);
    vec3 dNy = Color(V, albedo, roughness, N + dey)
             - Color(V, albedo, roughness, N - dey);
    vec3 dNz = Color(V, albedo, roughness, N + dez)
             - Color(V, albedo, roughness, N - dez);
    dNx /= (2 * delta);
    dNy /= (2 * delta);
    dNz /= (2 * delta);

//    FragColor = vec4(dNz, 1.0);
}
