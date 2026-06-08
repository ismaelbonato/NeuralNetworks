find_package(Protobuf REQUIRED)

set(PROTO_FILES
    ${CMAKE_CURRENT_SOURCE_DIR}/nn/model.proto
)

set(PROTO_GENERATED_DIR
    ${PROJECT_BINARY_DIR}/generated/proto
)

file(MAKE_DIRECTORY ${PROTO_GENERATED_DIR})

protobuf_generate(
    LANGUAGE cpp
    OUT_VAR PROTO_SRCS
    IMPORT_DIRS ${CMAKE_CURRENT_SOURCE_DIR}
    PROTOC_OUT_DIR ${PROTO_GENERATED_DIR}
    PROTOS ${PROTO_FILES}
)

set(PROTO_WRAPPER_SRCS
    ${PROJECT_SOURCE_DIR}/src/serialization/LayerSerialization.cpp
    ${PROJECT_SOURCE_DIR}/src/serialization/ModelSerialization.cpp
)

set(PROTO_LINT_SKIPPED_SRCS
    ${PROTO_SRCS}
    ${PROTO_WRAPPER_SRCS}
)

list(APPEND PROTO_SRCS
    ${PROTO_WRAPPER_SRCS}
)

set_source_files_properties(${PROTO_LINT_SKIPPED_SRCS}
    DIRECTORY ${PROJECT_SOURCE_DIR}
    PROPERTIES SKIP_LINTING TRUE
)

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    set_source_files_properties(${PROTO_SRCS}
        DIRECTORY ${PROJECT_SOURCE_DIR}
        PROPERTIES COMPILE_OPTIONS -Wno-sign-conversion
    )
endif()

add_custom_target(nn-runtime-protobuf-sources
    DEPENDS ${PROTO_SRCS}
)

add_library(nn-runtime-protobuf INTERFACE)
add_dependencies(nn-runtime-protobuf nn-runtime-protobuf-sources)

target_include_directories(nn-runtime-protobuf INTERFACE
    ${PROTO_GENERATED_DIR}
)

target_link_libraries(nn-runtime-protobuf INTERFACE
    protobuf::libprotobuf
)

set(PROTO_SRCS ${PROTO_SRCS} PARENT_SCOPE)
