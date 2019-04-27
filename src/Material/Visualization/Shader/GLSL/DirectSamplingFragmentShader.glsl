#version 330 core
out vec4 FragColor;

in vec3 position;
in vec3 normal;
in vec3 albedo;
in float roughness;
in float metallic;
in float ao;

// IBL
uniform samplerCube tex_env;
uniform samplerCube tex_env_diffuse;

uniform vec3 camera_position;

const float PI = 3.14159265359;

// ----------------------------------------------------------------------------
// http://holger.dammertz.org/stuff/notes_HammersleyOnHemisphere.html
// efficient VanDerCorpus calculation.
float RadicalInverse_VdC(uint bits) {
    bits = (bits << 16u) | (bits >> 16u);
    bits = ((bits & 0x55555555u) << 1u) | ((bits & 0xAAAAAAAAu) >> 1u);
    bits = ((bits & 0x33333333u) << 2u) | ((bits & 0xCCCCCCCCu) >> 2u);
    bits = ((bits & 0x0F0F0F0Fu) << 4u) | ((bits & 0xF0F0F0F0u) >> 4u);
    bits = ((bits & 0x00FF00FFu) << 8u) | ((bits & 0xFF00FF00u) >> 8u);
    return float(bits) * 2.3283064365386963e-10;// / 0x100000000
}
// ----------------------------------------------------------------------------
vec2 Hammersley(uint i, uint N) {
    return vec2(float(i)/float(N), RadicalInverse_VdC(i));
}
// ----------------------------------------------------------------------------
vec3 ImportanceSampleGGX(vec2 Xi, vec3 N, float roughness) {
    float a = roughness*roughness;

    float phi = 2.0 * PI * Xi.x;
    float cosTheta = sqrt((1.0 - Xi.y) / (1.0 + (a*a - 1.0) * Xi.y));
    float sinTheta = sqrt(1.0 - cosTheta*cosTheta);

    // from spherical coordinates to cartesian coordinates - halfway vector
    vec3 H;
    H.x = cos(phi) * sinTheta;
    H.y = sin(phi) * sinTheta;
    H.z = cosTheta;

    // from tangent-space H vector to world-space sample vector
    vec3 up        = abs(N.z) < 0.999 ? vec3(0.0, 0.0, 1.0) : vec3(1.0, 0.0, 0.0);
    vec3 tangent   = normalize(cross(up, N));
    vec3 bitangent = cross(N, tangent);

    vec3 sampleVec = tangent * H.x + bitangent * H.y + N * H.z;
    return normalize(sampleVec);
}

// ----------------------------------------------------------------------------
float GeometrySchlickGGX(float NdotV, float roughness)
{
    float r = (roughness + 1.0);
    float k = (r*r) / 8.0;

    float nom   = NdotV;
    float denom = NdotV * (1.0 - k) + k;

    return nom / denom;
}
// ----------------------------------------------------------------------------
float GeometrySmith(vec3 N, vec3 V, vec3 L, float roughness)
{
    float NdotV = max(dot(N, V), 0.0);
    float NdotL = max(dot(N, L), 0.0);
    float ggx2 = GeometrySchlickGGX(NdotV, roughness);
    float ggx1 = GeometrySchlickGGX(NdotL, roughness);

    return ggx1 * ggx2;
}
// ----------------------------------------------------------------------------
vec3 FresnelSchlick(float cosTheta, vec3 F0)
{
    return F0 + (1.0 - F0) * pow(1.0 - cosTheta, 5.0);
}
// ----------------------------------------------------------------------------
vec3 FresnelSchlickRoughness(float cosTheta, vec3 F0, float roughness)
{
    return F0 + (max(vec3(1.0 - roughness), F0) - F0) * pow(1.0 - cosTheta, 5.0);
}
// ----------------------------------------------------------------------------
void main()
{
    // material properties
    vec3 albedo_degamma = pow(albedo, vec3(2.2));

    // input lighting data
    vec3 N = normal;
    vec3 V = normalize(camera_position - position);

    // calculate reflectance at normal incidence; if dia-electric (like plastic) use F0
    // of 0.04 and if it's a metal, use the albedo color as F0 (metallic workflow)
    vec3 F0 = vec3(0.04);
    F0 = mix(F0, albedo_degamma, metallic);

    // ambient lighting (we now use IBL as the ambient term)
    vec3 F = FresnelSchlickRoughness(max(dot(N, V), 0.0), F0, roughness);
    vec3 kD = 1.0 - F;
    kD *= 1.0 - metallic;

    vec3 irradiance = texture(tex_env_diffuse, N).rgb;
    vec3 diffuse    = irradiance * albedo_degamma;

    vec3 specular = vec3(0.0);
    const uint num_samples = 1024u;
    for (uint i = 0u; i < num_samples; ++i) {
        vec2 Xi = Hammersley(i, num_samples);
        vec3 H = ImportanceSampleGGX(Xi, N, roughness);
        vec3 L = 2 * dot(V, H) * H - V;

        float NoV = clamp(dot(N, V), 0.0, 1.0);
        float NoL = clamp(dot(N, L), 0.0, 1.0);
        float NoH = clamp(dot(N, H), 0.0, 1.0);
        float VoH = clamp(dot(V, H), 0.0, 1.0);

        if (NoL > 0) {
            vec3 specular_sample = texture(tex_env, L).rgb;
            float G_sample = GeometrySmith(N, V, L, roughness);
            vec3  F_sample = FresnelSchlickRoughness(VoH, F0, roughness);
            specular += specular_sample * F_sample * G_sample * VoH / (NoH * NoV);
        }
    }
    specular /= num_samples;

    vec3 color = (kD * diffuse + specular) * ao;

    // HDR tonemapping
    color = color / (color + vec3(1.0));
    // gamma correct
    color = pow(color, vec3(1.0/2.2));

    FragColor = vec4(color, 1.0);
}
