project(tiny_obj_loader CXX)

cmake_minimum_required(VERSION 3.0.0)
set(CMAKE_ALLOW_LOOSE_LOOP_CONSTRUCTS true)

# SET LIBNAME
set(TINY_OBJ_LOADER_LIBRARY tiny_obj_loader)

file(GLOB tiny_obj_loader_sources tiny_obj_loader.cc)
file(GLOB tiny_obj_loader_headers tiny_obj_loader.h)

add_library(${TINY_OBJ_LOADER_LIBRARY} STATIC
        ${tiny_obj_loader_sources} ${tiny_obj_loader_headers})
set_target_properties(${TINY_OBJ_LOADER_LIBRARY} PROPERTIES
        OUTPUT_NAME ${TINY_OBJ_LOADER_LIBRARY}
        FOLDER "3rdparty")

# install
if (NOT BUILD_SHARED_LIBS)
    install(TARGETS ${TINY_OBJ_LOADER_LIBRARY}
            RUNTIME DESTINATION ${CMAKE_INSTALL_PREFIX}/bin
            LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX}/lib
            ARCHIVE DESTINATION ${CMAKE_INSTALL_PREFIX}/lib)
endif()
