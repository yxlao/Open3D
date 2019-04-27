file(GLOB_RECURSE MAT_ALL_HEADER_FILES "*.h")
file(GLOB_RECURSE MAT_ALL_SOURCE_FILES "*.cpp")
file(GLOB_RECURSE MAT_ALL_CUDA_HEADER_FILES "*.cuh")
file(GLOB_RECURSE MAT_ALL_CUDA_SOURCE_FILES "*.cu")

# some black magic to automatically create a header file from shader files
file(GLOB SHADER_FILES "Visualization/Shader/GLSL/*.glsl")

add_custom_command(
        OUTPUT ${PROJECT_SOURCE_DIR}/src/InverseRendering/Visualization/Shader/Shader.h
        COMMAND EncodeShader Shader.h
        WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}/src/InverseRendering/Visualization/Shader/
        DEPENDS ${SHADER_FILES} EncodeShader
)

foreach(file ${SHADER_FILES})
    add_custom_command(
            OUTPUT ${PROJECT_SOURCE_DIR}/src/InverseRendering/Visualization/Shader/Shader.h
            COMMAND EncodeShader Shader.h ${file}
            WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}/src/InverseRendering/Visualization/Shader/
            DEPENDS ${SHADER_FILES} EncodeShader
            APPEND
    )
endforeach(file)

add_custom_target(PBRShaderFileTarget ALL
        DEPENDS ${PROJECT_SOURCE_DIR}/src/InverseRendering/Visualization/Shader/Shader.h)
set_target_properties(PBRShaderFileTarget PROPERTIES FOLDER "Custom")

# build
file(GLOB_RECURSE VISUALIZATION_ALL_HEADER_FILES "*.h")
file(GLOB_RECURSE VISUALIZATION_ALL_SOURCE_FILES "*.cpp")

set(OPEN3D_MATERIAL_MODULE "Material" CACHE STRING "Material component library")
add_library(${OPEN3D_MATERIAL_MODULE}
        ${MAT_ALL_HEADER_FILES} ${MAT_ALL_SOURCE_FILES})
target_link_libraries(${OPEN3D_MATERIAL_MODULE}
        ${3RDPARTY_LIBRARIES} ${CMAKE_PROJECT_NAME})
target_include_directories(${OPEN3D_MATERIAL_MODULE} PUBLIC ${3RDPARTY_INCLUDE_DIRS})

source_group("Source Files\\Shader\\GLSL" FILES ${SHADER_FILES})
add_dependencies(Visualization PBRShaderFileTarget)
