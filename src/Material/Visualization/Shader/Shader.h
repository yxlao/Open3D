// Automatically generated header file for shader.
// See LICENSE.txt for full license statement.

#pragma once

// clang-format off
namespace open3d {

namespace visualization {

namespace glsl {

const char * const BackgroundFragmentShader = 
"#version 330 core\n"
"out vec4 FragColor;\n"
"in vec3 position;\n"
"\n"
"uniform samplerCube tex_env;\n"
"\n"
"void main() {\n"
"    vec3 envColor = textureLod(tex_env, position, 0.0).rgb;\n"
"\n"
"    // HDR tonemap and gamma correct\n"
"    envColor = envColor / (envColor + vec3(1.0));\n"
"    envColor = pow(envColor, vec3(1.0/2.2));\n"
"\n"
"    FragColor = vec4(envColor, 1.0);\n"
"}\n"
;

}  // namespace open3d::glsl

}  // namespace open3d::visualization

}  // namespace open3d

// clang-format on
// clang-format off
namespace open3d {

namespace visualization {

namespace glsl {

const char * const BackgroundVertexShader = 
"#version 330 core\n"
"\n"
"in vec3 vertex_position;\n"
"\n"
"uniform mat4 P;\n"
"uniform mat4 V;\n"
"\n"
"out vec3 position;\n"
"\n"
"void main() {\n"
"    position = vertex_position;\n"
"\n"
"	mat4 rotV = mat4(mat3(V));\n"
"	vec4 clipPos = P * rotV * vec4(position, 1.0);\n"
"\n"
"	gl_Position = clipPos.xyww;\n"
"}\n"
;

}  // namespace open3d::glsl

}  // namespace open3d::visualization

}  // namespace open3d

// clang-format on
// clang-format off
namespace open3d {

namespace visualization {

namespace glsl {

const char * const HDRToCubemapFragmentShader = 
"#version 330 core\n"
"out vec4 FragColor;\n"
"in vec3 position;\n"
"\n"
"uniform sampler2D tex_hdr;\n"
"\n"
"const vec2 invAtan = vec2(0.1591, 0.3183);\n"
"\n"
"vec2 SampleSphericalMap(vec3 v) {\n"
"    vec2 uv = vec2(atan(v.z, v.x), asin(v.y));\n"
"    uv *= invAtan;\n"
"    uv += 0.5;\n"
"    return uv;\n"
"}\n"
"\n"
"void main() {\n"
"    vec2 uv = SampleSphericalMap(normalize(position));\n"
"    vec3 color = texture(tex_hdr, uv).rgb;\n"
"\n"
"    FragColor = vec4(color, 1.0);\n"
"}\n"
;

}  // namespace open3d::glsl

}  // namespace open3d::visualization

}  // namespace open3d

// clang-format on
// clang-format off
namespace open3d {

namespace visualization {

namespace glsl {

const char * const IBLFragmentShader = 
"#version 330 core\n"
"out vec4 FragColor;\n"
"\n"
"in vec2 uv;\n"
"in vec3 position;\n"
"in vec3 normal;\n"
"\n"
"// material parameters\n"
"uniform sampler2D tex_albedo;\n"
"uniform sampler2D tex_normal;\n"
"uniform sampler2D tex_metallic;\n"
"uniform sampler2D tex_roughness;\n"
"uniform sampler2D tex_ao;\n"
"\n"
"// IBL\n"
"uniform samplerCube tex_env_diffuse;\n"
"uniform samplerCube tex_env_specular;\n"
"uniform sampler2D   tex_lut_specular;\n"
"\n"
"uniform vec3 camera_position;\n"
"\n"
"const float PI = 3.14159265359;\n"
"// ----------------------------------------------------------------------------\n"
"// Easy trick to get tangent-normals to world-space to keep PBR code simplified.\n"
"// Don't worry if you don't get what's going on; you generally want to do normal \n"
"// mapping the usual way for performance anways; I do plan make a note of this \n"
"// technique somewhere later in the normal mapping tutorial.\n"
"vec3 getnormalFromMap()\n"
"{\n"
"    vec3 tangentnormal = texture(tex_normal, uv).xyz * 2.0 - 1.0;\n"
"\n"
"    vec3 Q1  = dFdx(position);\n"
"    vec3 Q2  = dFdy(position);\n"
"    vec2 st1 = dFdx(uv);\n"
"    vec2 st2 = dFdy(uv);\n"
"\n"
"    vec3 N   = normalize(normal);\n"
"    vec3 T  = normalize(Q1*st2.t - Q2*st1.t);\n"
"    vec3 B  = -normalize(cross(N, T));\n"
"    mat3 TBN = mat3(T, B, N);\n"
"\n"
"    return normalize(TBN * tangentnormal);\n"
"}\n"
"// ----------------------------------------------------------------------------\n"
"float DistributionGGX(vec3 N, vec3 H, float roughness)\n"
"{\n"
"    float a = roughness*roughness;\n"
"    float a2 = a*a;\n"
"    float NdotH = max(dot(N, H), 0.0);\n"
"    float NdotH2 = NdotH*NdotH;\n"
"\n"
"    float nom   = a2;\n"
"    float denom = (NdotH2 * (a2 - 1.0) + 1.0);\n"
"    denom = PI * denom * denom;\n"
"\n"
"    return nom / denom;\n"
"}\n"
"// ----------------------------------------------------------------------------\n"
"float GeometrySchlickGGX(float NdotV, float roughness)\n"
"{\n"
"    float r = (roughness + 1.0);\n"
"    float k = (r*r) / 8.0;\n"
"\n"
"    float nom   = NdotV;\n"
"    float denom = NdotV * (1.0 - k) + k;\n"
"\n"
"    return nom / denom;\n"
"}\n"
"// ----------------------------------------------------------------------------\n"
"float GeometrySmith(vec3 N, vec3 V, vec3 L, float roughness)\n"
"{\n"
"    float NdotV = max(dot(N, V), 0.0);\n"
"    float NdotL = max(dot(N, L), 0.0);\n"
"    float ggx2 = GeometrySchlickGGX(NdotV, roughness);\n"
"    float ggx1 = GeometrySchlickGGX(NdotL, roughness);\n"
"\n"
"    return ggx1 * ggx2;\n"
"}\n"
"// ----------------------------------------------------------------------------\n"
"vec3 fresnelSchlick(float cosTheta, vec3 F0)\n"
"{\n"
"    return F0 + (1.0 - F0) * pow(1.0 - cosTheta, 5.0);\n"
"}\n"
"// ----------------------------------------------------------------------------\n"
"vec3 fresnelSchlickRoughness(float cosTheta, vec3 F0, float roughness)\n"
"{\n"
"    return F0 + (max(vec3(1.0 - roughness), F0) - F0) * pow(1.0 - cosTheta, 5.0);\n"
"}\n"
"// ----------------------------------------------------------------------------\n"
"void main()\n"
"{\n"
"    // material properties\n"
"    vec3  albedo = pow(texture(tex_albedo, uv).rgb, vec3(2.2));\n"
"    float metallic = texture(tex_metallic, uv).r;\n"
"    float roughness = texture(tex_roughness, uv).r;\n"
"    float ao = texture(tex_ao, uv).r;\n"
"\n"
"    // input lighting data\n"
"    vec3 N = getnormalFromMap();\n"
"    vec3 V = normalize(camera_position - position);\n"
"    vec3 R = reflect(-V, N);\n"
"\n"
"    // calculate reflectance at normal incidence; if dia-electric (like plastic) use F0\n"
"    // of 0.04 and if it's a metal, use the albedo color as F0 (metallic workflow)\n"
"    vec3 F0 = vec3(0.04);\n"
"    F0 = mix(F0, albedo, metallic);\n"
"\n"
"    // reflectance equation\n"
"    vec3 Lo = vec3(0.0);\n"
"\n"
"    // ambient lighting (we now use IBL as the ambient term)\n"
"    vec3 F = fresnelSchlickRoughness(max(dot(N, V), 0.0), F0, roughness);\n"
"\n"
"    vec3 kS = F;\n"
"    vec3 kD = 1.0 - kS;\n"
"    kD *= 1.0 - metallic;\n"
"\n"
"    vec3 irradiance = texture(tex_env_diffuse, N).rgb;\n"
"    vec3 diffuse    = irradiance * albedo;\n"
"\n"
"    // sample both the pre-filter map and the BRDF lut and combine them together as per the Split-Sum approximation to get the IBL specular part.\n"
"    const float MAX_REFLECTION_LOD = 4.0;\n"
"    vec3 prefilteredColor = textureLod(tex_env_specular, R, roughness * MAX_REFLECTION_LOD).rgb;\n"
"    vec2 brdf  = texture(tex_lut_specular, vec2(max(dot(N, V), 0.0), roughness)).rg;\n"
"    vec3 specular = prefilteredColor * (F * brdf.x + brdf.y);\n"
"\n"
"    vec3 ambient = (kD * diffuse + specular) * ao;\n"
"\n"
"    vec3 color = ambient + Lo;\n"
"\n"
"    // HDR tonemapping\n"
"    color = color / (color + vec3(1.0));\n"
"    // gamma correct\n"
"    color = pow(color, vec3(1.0/2.2));\n"
"\n"
"    FragColor = vec4(color, 1.0);\n"
"}\n"
;

}  // namespace open3d::glsl

}  // namespace open3d::visualization

}  // namespace open3d

// clang-format on
// clang-format off
namespace open3d {

namespace visualization {

namespace glsl {

const char * const IBLVertexShader = 
"#version 330 core\n"
"in vec3 vertex_position;\n"
"in vec2 vertex_uv;\n"
"in vec3 vertex_normal;\n"
"\n"
"out vec2 uv;\n"
"out vec3 position;\n"
"out vec3 normal;\n"
"\n"
"uniform mat4 P;\n"
"uniform mat4 V;\n"
"uniform mat4 M;\n"
"\n"
"void main() {\n"
"    uv = vertex_uv;\n"
"    position = vec3(M * vec4(vertex_position, 1.0));\n"
"    normal = mat3(M) * vertex_normal;\n"
"\n"
"    gl_Position =  P * V * vec4(position, 1.0);\n"
"}\n"
;

}  // namespace open3d::glsl

}  // namespace open3d::visualization

}  // namespace open3d

// clang-format on
// clang-format off
namespace open3d {

namespace visualization {

namespace glsl {

const char * const NoIBLFragmentShader = 
"#version 330 core\n"
"out vec4 FragColor;\n"
"\n"
"in vec3 position;\n"
"in vec3 normal;\n"
"in vec2 uv;\n"
"\n"
"// material parameters\n"
"uniform sampler2D tex_albedo;\n"
"uniform sampler2D tex_normal;\n"
"uniform sampler2D tex_metallic;\n"
"uniform sampler2D tex_roughness;\n"
"uniform sampler2D tex_ao;\n"
"\n"
"// lights\n"
"uniform vec3 light_positions[4];\n"
"uniform vec3 light_colors[4];\n"
"uniform vec3 camera_position;\n"
"\n"
"const float PI = 3.14159265359;\n"
"// ----------------------------------------------------------------------------\n"
"// Easy trick to get tangent-normals to world-space to keep PBR code simplified.\n"
"// Don't worry if you don't get what's going on; you generally want to do normal\n"
"// mapping the usual way for performance anways; I do plan make a note of this\n"
"// technique somewhere later in the normal mapping tutorial.\n"
"vec3 getnormalFromMap()\n"
"{\n"
"    vec3 tangentnormal = texture(tex_normal, uv).xyz * 2.0 - 1.0;\n"
"\n"
"    vec3 Q1  = dFdx(position);\n"
"    vec3 Q2  = dFdy(position);\n"
"    vec2 st1 = dFdx(uv);\n"
"    vec2 st2 = dFdy(uv);\n"
"\n"
"    vec3 N   = normalize(normal);\n"
"    vec3 T  = normalize(Q1*st2.t - Q2*st1.t);\n"
"    vec3 B  = -normalize(cross(N, T));\n"
"    mat3 TBN = mat3(T, B, N);\n"
"\n"
"    return normalize(TBN * tangentnormal);\n"
"}\n"
"// ----------------------------------------------------------------------------\n"
"float DistributionGGX(vec3 N, vec3 H, float roughness)\n"
"{\n"
"    float a = roughness*roughness;\n"
"    float a2 = a*a;\n"
"    float NdotH = max(dot(N, H), 0.0);\n"
"    float NdotH2 = NdotH*NdotH;\n"
"\n"
"    float nom   = a2;\n"
"    float denom = (NdotH2 * (a2 - 1.0) + 1.0);\n"
"    denom = PI * denom * denom;\n"
"\n"
"    return nom / denom;\n"
"}\n"
"// ----------------------------------------------------------------------------\n"
"float GeometrySchlickGGX(float NdotV, float roughness)\n"
"{\n"
"    float r = (roughness + 1.0);\n"
"    float k = (r*r) / 8.0;\n"
"\n"
"    float nom   = NdotV;\n"
"    float denom = NdotV * (1.0 - k) + k;\n"
"\n"
"    return nom / denom;\n"
"}\n"
"// ----------------------------------------------------------------------------\n"
"float GeometrySmith(vec3 N, vec3 V, vec3 L, float roughness)\n"
"{\n"
"    float NdotV = max(dot(N, V), 0.0);\n"
"    float NdotL = max(dot(N, L), 0.0);\n"
"    float ggx2 = GeometrySchlickGGX(NdotV, roughness);\n"
"    float ggx1 = GeometrySchlickGGX(NdotL, roughness);\n"
"\n"
"    return ggx1 * ggx2;\n"
"}\n"
"// ----------------------------------------------------------------------------\n"
"vec3 fresnelSchlick(float cosTheta, vec3 F0)\n"
"{\n"
"    return F0 + (1.0 - F0) * pow(1.0 - cosTheta, 5.0);\n"
"}\n"
"// ----------------------------------------------------------------------------\n"
"void main()\n"
"{\n"
"    vec3 albedo     = pow(texture(tex_albedo, uv).rgb, vec3(2.2));\n"
"    float metallic  = texture(tex_metallic, uv).r;\n"
"    float roughness = texture(tex_roughness, uv).r;\n"
"    float ao        = texture(tex_ao, uv).r;\n"
"\n"
"    vec3 N = getnormalFromMap();\n"
"    vec3 V = normalize(camera_position - position);\n"
"\n"
"    // calculate reflectance at normal incidence; if dia-electric (like plastic) use F0\n"
"    // of 0.04 and if it's a metal, use the albedo color as F0 (metallic workflow)\n"
"    vec3 F0 = vec3(0.04);\n"
"    F0 = mix(F0, albedo, metallic);\n"
"\n"
"    // reflectance equation\n"
"    vec3 Lo = vec3(0.0);\n"
"    for(int i = 0; i < 4; ++i) {\n"
"        // calculate per-light radiance\n"
"        vec3 L = normalize(light_positions[i] - position);\n"
"        vec3 H = normalize(V + L);\n"
"        float distance = length(light_positions[i] - position);\n"
"        float attenuation = 1.0 / (distance * distance);\n"
"        vec3 radiance = light_colors[i] * attenuation;\n"
"\n"
"        // Cook-Torrance BRDF\n"
"        float NDF = DistributionGGX(N, H, roughness);\n"
"        float G   = GeometrySmith(N, V, L, roughness);\n"
"        vec3 F    = fresnelSchlick(max(dot(H, V), 0.0), F0);\n"
"\n"
"        vec3 nominator    = NDF * G * F;\n"
"\n"
"        // 0.001 to prevent divide by zero.\n"
"        float denominator = 4 * max(dot(N, V), 0.0) * max(dot(N, L), 0.0) + 0.001;\n"
"        vec3 specular = nominator / denominator;\n"
"\n"
"        // kS is equal to Fresnel\n"
"        vec3 kS = F;\n"
"        // for energy conservation, the diffuse and specular light can't\n"
"        // be above 1.0 (unless the surface emits light); to preserve this\n"
"        // relationship the diffuse component (kD) should equal 1.0 - kS.\n"
"        vec3 kD = vec3(1.0) - kS;\n"
"        // multiply kD by the inverse metalness such that only non-metals\n"
"        // have diffuse lighting, or a linear blend if partly metal (pure metals\n"
"        // have no diffuse light).\n"
"        kD *= 1.0 - metallic;\n"
"\n"
"        // scale light by NdotL\n"
"        float NdotL = max(dot(N, L), 0.0);\n"
"\n"
"        // add to outgoing radiance Lo\n"
"        // note that we already multiplied the BRDF by the Fresnel (kS) so we won't multiply by kS again\n"
"        Lo += (kD * albedo / PI + specular) * radiance * NdotL;\n"
"    }\n"
"\n"
"    // ambient lighting (note that the next IBL tutorial will replace\n"
"    // this ambient lighting with environment lighting).\n"
"    vec3 ambient = vec3(0.03) * albedo * ao;\n"
"\n"
"    vec3 color = ambient + Lo;\n"
"\n"
"    // HDR tonemapping\n"
"    color = color / (color + vec3(1.0));\n"
"    // gamma correct\n"
"    color = pow(color, vec3(1.0/2.2));\n"
"\n"
"    FragColor = vec4(color, 1.0);\n"
"}\n"
;

}  // namespace open3d::glsl

}  // namespace open3d::visualization

}  // namespace open3d

// clang-format on
// clang-format off
namespace open3d {

namespace visualization {

namespace glsl {

const char * const NoIBLVertexShader = 
"#version 330 core\n"
"\n"
"in vec3 vertex_position;\n"
"in vec3 vertex_normal;\n"
"in vec2 vertex_uv;\n"
"\n"
"out vec3 position;\n"
"out vec3 normal;\n"
"out vec2 uv;\n"
"\n"
"uniform mat4 M;\n"
"uniform mat4 V;\n"
"uniform mat4 P;\n"
"\n"
"void main() {\n"
"    uv = vertex_uv;\n"
"    position = vec3(M * vec4(vertex_position, 1.0));\n"
"    normal = mat3(M) * vertex_normal;\n"
"\n"
"    gl_Position = P * V * vec4(position, 1.0);\n"
"}\n"
;

}  // namespace open3d::glsl

}  // namespace open3d::visualization

}  // namespace open3d

// clang-format on
// clang-format off
namespace open3d {

namespace visualization {

namespace glsl {

const char * const PreConvDiffuseFragmentShader = 
"#version 330 core\n"
"out vec4 FragColor;\n"
"in vec3 position;\n"
"\n"
"uniform samplerCube tex_env;\n"
"\n"
"const float PI = 3.14159265359;\n"
"\n"
"void main() {\n"
"    vec3 N = normalize(position);\n"
"\n"
"    vec3 irradiance = vec3(0.0);   \n"
"    \n"
"    // tangent space calculation from origin point\n"
"    vec3 up    = vec3(0.0, 1.0, 0.0);\n"
"    vec3 right = cross(up, N);\n"
"    up         = cross(N, right);\n"
"       \n"
"    float sampleDelta = 0.025;\n"
"    float nrSamples = 0.0f;\n"
"    for(float phi = 0.0; phi < 2.0 * PI; phi += sampleDelta) {\n"
"        for(float theta = 0.0; theta < 0.5 * PI; theta += sampleDelta) {\n"
"            // spherical to cartesian (in tangent space)\n"
"            vec3 tangentSample = vec3(sin(theta) * cos(phi),  sin(theta) * sin(phi), cos(theta));\n"
"            // tangent space to world\n"
"            vec3 sampleVec = tangentSample.x * right + tangentSample.y * up + tangentSample.z * N; \n"
"\n"
"            irradiance += texture(tex_env, sampleVec).rgb * cos(theta) * sin(theta);\n"
"            nrSamples++;\n"
"        }\n"
"    }\n"
"    irradiance = PI * irradiance * (1.0 / float(nrSamples));\n"
"    \n"
"    FragColor = vec4(irradiance, 1.0);\n"
"}\n"
;

}  // namespace open3d::glsl

}  // namespace open3d::visualization

}  // namespace open3d

// clang-format on
// clang-format off
namespace open3d {

namespace visualization {

namespace glsl {

const char * const PreFilterLightingFragmentShader = 
"#version 330 core\n"
"out vec4 FragColor;\n"
"in vec3 position;\n"
"\n"
"uniform samplerCube tex_env;\n"
"uniform float roughness;\n"
"\n"
"const float PI = 3.14159265359;\n"
"// ----------------------------------------------------------------------------\n"
"float DistributionGGX(vec3 N, vec3 H, float roughness)\n"
"{\n"
"    float a = roughness*roughness;\n"
"    float a2 = a*a;\n"
"    float NdotH = max(dot(N, H), 0.0);\n"
"    float NdotH2 = NdotH*NdotH;\n"
"\n"
"    float nom   = a2;\n"
"    float denom = (NdotH2 * (a2 - 1.0) + 1.0);\n"
"    denom = PI * denom * denom;\n"
"\n"
"    return nom / denom;\n"
"}\n"
"// ----------------------------------------------------------------------------\n"
"// http://holger.dammertz.org/stuff/notes_HammersleyOnHemisphere.html\n"
"// efficient VanDerCorpus calculation.\n"
"float RadicalInverse_VdC(uint bits)\n"
"{\n"
"    bits = (bits << 16u) | (bits >> 16u);\n"
"    bits = ((bits & 0x55555555u) << 1u) | ((bits & 0xAAAAAAAAu) >> 1u);\n"
"    bits = ((bits & 0x33333333u) << 2u) | ((bits & 0xCCCCCCCCu) >> 2u);\n"
"    bits = ((bits & 0x0F0F0F0Fu) << 4u) | ((bits & 0xF0F0F0F0u) >> 4u);\n"
"    bits = ((bits & 0x00FF00FFu) << 8u) | ((bits & 0xFF00FF00u) >> 8u);\n"
"    return float(bits) * 2.3283064365386963e-10;// / 0x100000000\n"
"}\n"
"// ----------------------------------------------------------------------------\n"
"vec2 Hammersley(uint i, uint N)\n"
"{\n"
"    return vec2(float(i)/float(N), RadicalInverse_VdC(i));\n"
"}\n"
"// ----------------------------------------------------------------------------\n"
"vec3 ImportanceSampleGGX(vec2 Xi, vec3 N, float roughness)\n"
"{\n"
"    float a = roughness*roughness;\n"
"\n"
"    float phi = 2.0 * PI * Xi.x;\n"
"    float cosTheta = sqrt((1.0 - Xi.y) / (1.0 + (a*a - 1.0) * Xi.y));\n"
"    float sinTheta = sqrt(1.0 - cosTheta*cosTheta);\n"
"\n"
"    // from spherical coordinates to cartesian coordinates - halfway vector\n"
"    vec3 H;\n"
"    H.x = cos(phi) * sinTheta;\n"
"    H.y = sin(phi) * sinTheta;\n"
"    H.z = cosTheta;\n"
"\n"
"    // from tangent-space H vector to world-space sample vector\n"
"    vec3 up          = abs(N.z) < 0.999 ? vec3(0.0, 0.0, 1.0) : vec3(1.0, 0.0, 0.0);\n"
"    vec3 tangent   = normalize(cross(up, N));\n"
"    vec3 bitangent = cross(N, tangent);\n"
"\n"
"    vec3 sampleVec = tangent * H.x + bitangent * H.y + N * H.z;\n"
"    return normalize(sampleVec);\n"
"}\n"
"// ----------------------------------------------------------------------------\n"
"void main() {\n"
"    vec3 N = normalize(position);\n"
"\n"
"    // make the simplyfying assumption that V equals R equals the normal \n"
"    vec3 R = N;\n"
"    vec3 V = R;\n"
"\n"
"    const uint SAMPLE_COUNT = 1024u;\n"
"    vec3 prefilteredColor = vec3(0.0);\n"
"    float totalWeight = 0.0;\n"
"\n"
"    for (uint i = 0u; i < SAMPLE_COUNT; ++i) {\n"
"        // generates a sample vector that's biased towards the preferred alignment direction (importance sampling).\n"
"        vec2 Xi = Hammersley(i, SAMPLE_COUNT);\n"
"        vec3 H = ImportanceSampleGGX(Xi, N, roughness);\n"
"        vec3 L  = normalize(2.0 * dot(V, H) * H - V);\n"
"\n"
"        float NdotL = max(dot(N, L), 0.0);\n"
"        if (NdotL > 0.0) {\n"
"            // sample from the environment's mip level based on roughness/pdf\n"
"            float D   = DistributionGGX(N, H, roughness);\n"
"            float NdotH = max(dot(N, H), 0.0);\n"
"            float HdotV = max(dot(H, V), 0.0);\n"
"            float pdf = D * NdotH / (4.0 * HdotV) + 0.0001;\n"
"\n"
"            float resolution = 512.0;// resolution of source cubemap (per face)\n"
"            float saTexel  = 4.0 * PI / (6.0 * resolution * resolution);\n"
"            float saSample = 1.0 / (float(SAMPLE_COUNT) * pdf + 0.0001);\n"
"\n"
"            float mipLevel = roughness == 0.0 ? 0.0 : 0.5 * log2(saSample / saTexel);\n"
"\n"
"            prefilteredColor += textureLod(tex_env, L, mipLevel).rgb * NdotL;\n"
"            totalWeight      += NdotL;\n"
"        }\n"
"    }\n"
"\n"
"    prefilteredColor = prefilteredColor / totalWeight;\n"
"\n"
"    FragColor = vec4(prefilteredColor, 1.0);\n"
"}\n"
;

}  // namespace open3d::glsl

}  // namespace open3d::visualization

}  // namespace open3d

// clang-format on
// clang-format off
namespace open3d {

namespace visualization {

namespace glsl {

const char * const PreIntegrateLUTFragmentShader = 
"#version 330 core\n"
"\n"
"out vec2 FragColor;\n"
"in vec2 uv;\n"
"\n"
"const float PI = 3.14159265359;\n"
"// ----------------------------------------------------------------------------\n"
"// http://holger.dammertz.org/stuff/notes_HammersleyOnHemisphere.html\n"
"// efficient VanDerCorpus calculation.\n"
"float RadicalInverse_VdC(uint bits) \n"
"{\n"
"     bits = (bits << 16u) | (bits >> 16u);\n"
"     bits = ((bits & 0x55555555u) << 1u) | ((bits & 0xAAAAAAAAu) >> 1u);\n"
"     bits = ((bits & 0x33333333u) << 2u) | ((bits & 0xCCCCCCCCu) >> 2u);\n"
"     bits = ((bits & 0x0F0F0F0Fu) << 4u) | ((bits & 0xF0F0F0F0u) >> 4u);\n"
"     bits = ((bits & 0x00FF00FFu) << 8u) | ((bits & 0xFF00FF00u) >> 8u);\n"
"     return float(bits) * 2.3283064365386963e-10; // / 0x100000000\n"
"}\n"
"// ----------------------------------------------------------------------------\n"
"vec2 Hammersley(uint i, uint N)\n"
"{\n"
"	return vec2(float(i)/float(N), RadicalInverse_VdC(i));\n"
"}\n"
"// ----------------------------------------------------------------------------\n"
"vec3 ImportanceSampleGGX(vec2 Xi, vec3 N, float roughness)\n"
"{\n"
"	float a = roughness*roughness;\n"
"	\n"
"	float phi = 2.0 * PI * Xi.x;\n"
"	float cosTheta = sqrt((1.0 - Xi.y) / (1.0 + (a*a - 1.0) * Xi.y));\n"
"	float sinTheta = sqrt(1.0 - cosTheta*cosTheta);\n"
"	\n"
"	// from spherical coordinates to cartesian coordinates - halfway vector\n"
"	vec3 H;\n"
"	H.x = cos(phi) * sinTheta;\n"
"	H.y = sin(phi) * sinTheta;\n"
"	H.z = cosTheta;\n"
"	\n"
"	// from tangent-space H vector to world-space sample vector\n"
"	vec3 up          = abs(N.z) < 0.999 ? vec3(0.0, 0.0, 1.0) : vec3(1.0, 0.0, 0.0);\n"
"	vec3 tangent   = normalize(cross(up, N));\n"
"	vec3 bitangent = cross(N, tangent);\n"
"	\n"
"	vec3 sampleVec = tangent * H.x + bitangent * H.y + N * H.z;\n"
"	return normalize(sampleVec);\n"
"}\n"
"// ----------------------------------------------------------------------------\n"
"float GeometrySchlickGGX(float NdotV, float roughness)\n"
"{\n"
"    // note that we use a different k for IBL\n"
"    float a = roughness;\n"
"    float k = (a * a) / 2.0;\n"
"\n"
"    float nom   = NdotV;\n"
"    float denom = NdotV * (1.0 - k) + k;\n"
"\n"
"    return nom / denom;\n"
"}\n"
"// ----------------------------------------------------------------------------\n"
"float GeometrySmith(vec3 N, vec3 V, vec3 L, float roughness)\n"
"{\n"
"    float NdotV = max(dot(N, V), 0.0);\n"
"    float NdotL = max(dot(N, L), 0.0);\n"
"    float ggx2 = GeometrySchlickGGX(NdotV, roughness);\n"
"    float ggx1 = GeometrySchlickGGX(NdotL, roughness);\n"
"\n"
"    return ggx1 * ggx2;\n"
"}\n"
"// ----------------------------------------------------------------------------\n"
"vec2 IntegrateBRDF(float NdotV, float roughness)\n"
"{\n"
"    vec3 V;\n"
"    V.x = sqrt(1.0 - NdotV*NdotV);\n"
"    V.y = 0.0;\n"
"    V.z = NdotV;\n"
"\n"
"    float A = 0.0;\n"
"    float B = 0.0; \n"
"\n"
"    vec3 N = vec3(0.0, 0.0, 1.0);\n"
"    \n"
"    const uint SAMPLE_COUNT = 1024u;\n"
"    for(uint i = 0u; i < SAMPLE_COUNT; ++i)\n"
"    {\n"
"        // generates a sample vector that's biased towards the\n"
"        // preferred alignment direction (importance sampling).\n"
"        vec2 Xi = Hammersley(i, SAMPLE_COUNT);\n"
"        vec3 H = ImportanceSampleGGX(Xi, N, roughness);\n"
"        vec3 L = normalize(2.0 * dot(V, H) * H - V);\n"
"\n"
"        float NdotL = max(L.z, 0.0);\n"
"        float NdotH = max(H.z, 0.0);\n"
"        float VdotH = max(dot(V, H), 0.0);\n"
"\n"
"        if(NdotL > 0.0)\n"
"        {\n"
"            float G = GeometrySmith(N, V, L, roughness);\n"
"            float G_Vis = (G * VdotH) / (NdotH * NdotV);\n"
"            float Fc = pow(1.0 - VdotH, 5.0);\n"
"\n"
"            A += (1.0 - Fc) * G_Vis;\n"
"            B += Fc * G_Vis;\n"
"        }\n"
"    }\n"
"    A /= float(SAMPLE_COUNT);\n"
"    B /= float(SAMPLE_COUNT);\n"
"    return vec2(A, B);\n"
"}\n"
"// ----------------------------------------------------------------------------\n"
"void main() \n"
"{\n"
"    vec2 integratedBRDF = IntegrateBRDF(uv.x, uv.y);\n"
"    FragColor = integratedBRDF;\n"
"}\n"
;

}  // namespace open3d::glsl

}  // namespace open3d::visualization

}  // namespace open3d

// clang-format on
// clang-format off
namespace open3d {

namespace visualization {

namespace glsl {

const char * const PreIntegrateLUTVertexShader = 
"#version 330 core\n"
"\n"
"in vec3 vertex_position;\n"
"in vec2 vertex_uv;\n"
"\n"
"out vec2 uv;\n"
"\n"
"void main()\n"
"{\n"
"    uv = vertex_uv;\n"
"	gl_Position = vec4(vertex_position, 1.0);\n"
"}\n"
;

}  // namespace open3d::glsl

}  // namespace open3d::visualization

}  // namespace open3d

// clang-format on
// clang-format off
namespace open3d {

namespace visualization {

namespace glsl {

const char * const SimpleVertexShader = 
"#version 330 core\n"
"\n"
"in vec3 vertex_position;\n"
"\n"
"out vec3 position;\n"
"\n"
"uniform mat4 P;\n"
"uniform mat4 V;\n"
"\n"
"void main() {\n"
"    position = vertex_position;  \n"
"    gl_Position =  P * V * vec4(position, 1.0);\n"
"}\n"
;

}  // namespace open3d::glsl

}  // namespace open3d::visualization

}  // namespace open3d

// clang-format on
