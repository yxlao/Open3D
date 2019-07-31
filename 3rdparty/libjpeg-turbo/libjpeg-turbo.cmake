include(ExternalProject)

# Set compiler flags
if ("${CMAKE_C_COMPILER_ID}" STREQUAL "MSVC")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} /D_CRT_SECURE_NO_WARNINGS")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /D_CRT_SECURE_NO_WARNINGS")
endif()

# Set WITH_SIMD
include(CheckLanguage)
check_language(ASM_NASM)
if (CMAKE_ASM_NASM_COMPILER)
    if (APPLE)
        # macOS might have /usr/bin/nasm but it cannot be used
        # https://stackoverflow.com/q/53974320
        # To fix this, run `brew install nasm`
        execute_process(COMMAND nasm --version RESULT_VARIABLE return_code)
        if("${return_code}" STREQUAL "0")
            enable_language(ASM_NASM)
            option(WITH_SIMD "" ON)
        else()
            message(STATUS "nasm found but can't be used, run `brew install nasm`")
            option(WITH_SIMD "" OFF)
        endif()
    else()
        enable_language(ASM_NASM)
        option(WITH_SIMD "" ON)
    endif()
else()
    option(WITH_SIMD "" OFF)
endif()
if (WITH_SIMD)
    message(STATUS "NASM assembler enabled")
else()
    message(WARNING "NASM assembler not found - libjpeg-turbo performance may suffer")
endif()

ExternalProject_Add(
    ext_turbojpeg
    PREFIX turbojpeg
    SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/libjpeg-turbo/libjpeg-turbo
    UPDATE_COMMAND ""
    CMAKE_GENERATOR ${CMAKE_GENERATOR}
    CMAKE_GENERATOR_PLATFORM ${CMAKE_GENERATOR_PLATFORM}
    CMAKE_GENERATOR_TOOLSET ${CMAKE_GENERATOR_TOOLSET}
    CMAKE_ARGS
        -DCMAKE_CXX_FLAGS=${CMAKE_CXX_FLAGS}
        -DCMAKE_C_FLAGS=${DCMAKE_C_FLAGS}
        -DENABLE_STATIC=ON
        -DWITH_SIMD=${WITH_SIMD}
        -DCMAKE_INSTALL_PREFIX=${CMAKE_BINARY_DIR}
        -DCMAKE_POSITION_INDEPENDENT_CODE=ON
)

# This generates turbojpeg-static target
add_library(turbojpeg INTERFACE)
add_dependencies(turbojpeg ext_turbojpeg)

ExternalProject_Get_Property(ext_turbojpeg SOURCE_DIR BINARY_DIR)

set(turbojpeg_LIB_PATH ${CMAKE_BINARY_DIR}/lib/${CMAKE_STATIC_LIBRARY_PREFIX}turbojpeg${CMAKE_STATIC_LIBRARY_SUFFIX})
target_include_directories(turbojpeg SYSTEM INTERFACE
    ${CMAKE_BINARY_DIR}/include
)
target_link_libraries(turbojpeg INTERFACE
    ${turbojpeg_LIB_PATH}
)
set(JPEG_TURBO_LIBRARIES turbojpeg)

if (NOT BUILD_SHARED_LIBS)
    install(
        FILES
            ${turbojpeg_LIB_PATH}
        DESTINATION
            ${CMAKE_INSTALL_PREFIX}/lib
    )
endif()


