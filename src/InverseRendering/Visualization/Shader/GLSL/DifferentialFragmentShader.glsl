#version 330 core

layout(location = 0) out vec3 FragColor;

layout(location = 1) out vec3 residual;

layout(location = 2) out vec3 grad_albedo;
layout(location = 3) out vec3 grad_material;


//layout(location = 3) out vec3 dNx;
//layout(location = 4) out vec3 dNy;
//layout(location = 5) out vec3 dNz;


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

uniform sampler2D   tex_target_image;

uniform vec3 camera_position;
uniform vec2 viewport;

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

vec3 Color(vec3 V, vec3 albedo, vec3 material, vec3 N) {
    vec3 albedo_degamma = pow(albedo + 0.001, vec3(2.2));

    float roughness = material.r;
    float metallic = material.g;
    float ao = material.b;

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

    // Re-pack
    vec3 material = vec3(roughness, metallic, ao);

    // Renderering
    vec3 color = Color(V, albedo, material, N);
    FragColor = color;

    /** residual **/
    vec2 uv = gl_FragCoord.xy / viewport;
    residual = color - texture(tex_target_image, uv).rgb;

    /** output 1: gradient color **/
    const float delta = 0.001f;
    vec3 delta_p_rgb = min(1 - albedo, delta);
    vec3 delta_m_rgb = min(albedo, delta);
    vec3 dR = Color(V, albedo + vec3(delta_p_rgb.r, 0, 0), material, N)
            - Color(V, albedo - vec3(delta_m_rgb.r, 0, 0), material, N);
    vec3 dG = Color(V, albedo + vec3(0, delta_p_rgb.g, 0), material, N)
            - Color(V, albedo - vec3(0, delta_m_rgb.g, 0), material, N);
    vec3 dB = Color(V, albedo + vec3(0, 0, delta_p_rgb.b), material, N)
            - Color(V, albedo - vec3(0, 0, delta_m_rgb.b), material, N);
    vec3 d_albedo = vec3(dR.r, dG.g, dB.b) / (delta_p_rgb + delta_m_rgb);
    grad_albedo = d_albedo * residual;

    /** output 2: gradeint material **/
    vec3 delta_p_mat = min(1 - material, delta);
    vec3 delta_m_mat = min(material, delta);
    vec3 d_roughness = Color(V, albedo, material + vec3(delta_p_mat.r, 0, 0), N);
                     - Color(V, albedo, material - vec3(delta_m_mat.r, 0, 0), N);
    vec3 d_metallic  = Color(V, albedo, material + vec3(0, delta_p_mat.g, 0), N);
                     - Color(V, albedo, material - vec3(0, delta_m_mat.g, 0), N);
    vec3 d_ao        = Color(V, albedo, material + vec3(0, 0, delta_p_mat.b), N);
                     - Color(V, albedo, material - vec3(0, 0, delta_m_mat.b), N);
    grad_material = vec3(dot(d_roughness, residual),
                         dot(d_metallic, residual),
                         dot(d_ao, residual))
                     / (delta_p_mat + delta_m_mat);

    // so3 difference
    // [ 1   -dz   dy]
    // [ dz   1   -dx]
    // [-dy   dx   1 ]
//    vec3 dex = delta * vec3(0, -N.z, N.y);
//    vec3 dey = delta * vec3(N.z, 0, -N.x);
//    vec3 dez = delta * vec3(-N.y, N.x, 0);
//    dNx = Color(V, albedo, roughness, N + dex)
//        - Color(V, albedo, roughness, N - dex);
//    dNy = Color(V, albedo, roughness, N + dey)
//        - Color(V, albedo, roughness, N - dey);
//    dNz = Color(V, albedo, roughness, N + dez)
//        - Color(V, albedo, roughness, N - dez);
//    dNx /= (2 * delta);
//    dNy /= (2 * delta);
//    dNz /= (2 * delta);
}
