cmake_minimum_required(VERSION 3.20)

project(nn-runtime)

include(CTest)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Debug CACHE STRING "Build type" FORCE)
endif()

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(cmake/Warnings.cmake)
include(cmake/StaticAnalysis.cmake)
include(cmake/Coverage.cmake)

option(ENABLE_PROTOBUF "Build protobuf model serialization support" ON)

if(ENABLE_PROTOBUF)
    find_package(Protobuf REQUIRED)
    add_subdirectory(proto)

    set(PROTO_WRAPPER_SRCS
        src/serialization/ModelSerialization.cpp
    )

    set(PROTO_LINT_SKIPPED_SRCS
        ${PROTO_SRCS}
        ${PROTO_WRAPPER_SRCS}
    )

    list(APPEND PROTO_SRCS
        ${PROTO_WRAPPER_SRCS}
    )

    set_source_files_properties(${PROTO_LINT_SKIPPED_SRCS} PROPERTIES
        SKIP_LINTING TRUE
    )

    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        set_source_files_properties(${PROTO_SRCS} PROPERTIES
            COMPILE_OPTIONS -Wno-sign-conversion
        )
    endif()
endif()

set(SOURCES
    src/base/Layer.cpp
    src/base/Model.cpp
    src/layers/ConvolutionalLayer.cpp
    src/layers/DenseLayer.cpp
    src/layers/FlattenLayer.cpp
    src/layers/HopfieldLayer.cpp
    ${PROTO_SRCS}
)

set(HEADERS
    include/base/ActivationFunction.h
    include/base/ConvolutionGeometry.h
    include/base/Layer.h
    include/base/Model.h
    include/base/Parameters.h
    include/base/Tensor.h
    include/base/Types.h
    include/layers/ConvolutionalLayer.h
    include/layers/DenseLayer.h
    include/layers/FlattenLayer.h
    include/layers/HopfieldLayer.h
)

set(CPPCHECK_SOURCES ${SOURCES})
if(ENABLE_PROTOBUF)
    list(REMOVE_ITEM CPPCHECK_SOURCES ${PROTO_SRCS})
endif()

network_add_cppcheck_target(cppcheck
    SOURCES ${CPPCHECK_SOURCES}
    HEADERS ${HEADERS}
)

add_library(${PROJECT_NAME}
    ${SOURCES}
    ${HEADERS}
)

if(ENABLE_PROTOBUF)
    add_dependencies(${PROJECT_NAME} nn-runtime-protobuf-sources)
endif()

target_include_directories(${PROJECT_NAME} PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

if(ENABLE_PROTOBUF)
    target_include_directories(${PROJECT_NAME} PRIVATE
        ${CMAKE_CURRENT_BINARY_DIR}/generated/proto
    )

    target_link_libraries(${PROJECT_NAME} PRIVATE
        protobuf::libprotobuf
    )
endif()

network_enable_warnings(${PROJECT_NAME})

if(PROJECT_NAME STREQUAL CMAKE_PROJECT_NAME)
    set(MAIN_EXECUTABLE "${PROJECT_NAME}-main")

    add_executable(${MAIN_EXECUTABLE} main.cpp)

    target_link_libraries(${MAIN_EXECUTABLE} PRIVATE
        ${PROJECT_NAME}
    )

    target_include_directories(${MAIN_EXECUTABLE} PRIVATE
        ${CMAKE_SOURCE_DIR}/include
    )

    network_enable_warnings(${MAIN_EXECUTABLE})

    if(BUILD_TESTING)
        add_subdirectory(tests)
    endif()
endif()
