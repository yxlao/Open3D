project(stb_image C)

cmake_minimum_required(VERSION 3.0.0)
set(CMAKE_ALLOW_LOOSE_LOOP_CONSTRUCTS true)

# SET LIBNAME
set(STB_IMAGE_LIBRARY stb_image)

file(GLOB stb_image_sources stb_image.c stb_image_write.c)
file(GLOB stb_image_headers stb_image.h stb_image_write.h)

add_library(${STB_IMAGE_LIBRARY} STATIC ${stb_image_sources} ${stb_image_headers})
set_target_properties(${STB_IMAGE_LIBRARY} PROPERTIES
        OUTPUT_NAME ${STB_IMAGE_LIBRARY}
        FOLDER "3rdparty"
	)

# install
if (NOT BUILD_SHARED_LIBS)
    install(TARGETS ${STB_IMAGE_LIBRARY}
            RUNTIME DESTINATION ${CMAKE_INSTALL_PREFIX}/bin
            LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX}/lib
            ARCHIVE DESTINATION ${CMAKE_INSTALL_PREFIX}/lib)
endif()
