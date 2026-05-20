cmake_minimum_required(VERSION 3.20)

project(nn-runtime)

include(CTest)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Debug CACHE STRING "Build type" FORCE)
endif()

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(ENABLE_CPPCHECK "Run cppcheck as part of C++ builds" ON)
option(ENABLE_CLANG_TIDY "Run clang-tidy as part of C++ builds" ON)
option(ENABLE_COVERAGE "Build with coverage instrumentation and add a coverage target" OFF)
option(ENABLE_PROTOBUF "Build protobuf model serialization support" ON)

function(network_enable_warnings target)
    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        target_compile_options(${target} PRIVATE
            -Wall
            -Wextra
            -Wpedantic
            -Wshadow
            -Wconversion
            -Wsign-conversion
            -Wold-style-cast
            -Wnon-virtual-dtor
        )
    endif()
endfunction()

find_program(CPPCHECK_EXECUTABLE NAMES cppcheck)
set(CPPCHECK_ARGS
    --enable=warning,style,performance,portability
    --inline-suppr
    --language=c++
    --std=c++23
    --suppress=missingIncludeSystem
)

if(ENABLE_CPPCHECK)
    if(NOT CPPCHECK_EXECUTABLE)
        message(FATAL_ERROR "ENABLE_CPPCHECK is ON, but cppcheck was not found")
    endif()

    set(CMAKE_CXX_CPPCHECK
        ${CPPCHECK_EXECUTABLE}
        ${CPPCHECK_ARGS}
    )
endif()

find_program(CLANG_TIDY_EXECUTABLE NAMES clang-tidy-20 clang-tidy)
set(CLANG_TIDY_ARGS
    --extra-arg=-std=c++23
)

if(ENABLE_CLANG_TIDY)
    if(NOT CLANG_TIDY_EXECUTABLE)
        message(FATAL_ERROR "ENABLE_CLANG_TIDY is ON, but clang-tidy was not found")
    endif()

    set(CMAKE_CXX_CLANG_TIDY
        ${CLANG_TIDY_EXECUTABLE}
        ${CLANG_TIDY_ARGS}
    )
endif()

if(ENABLE_COVERAGE)
    if(NOT CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        message(FATAL_ERROR "ENABLE_COVERAGE requires GCC or Clang")
    endif()

    find_program(GCOVR_EXECUTABLE NAMES gcovr)
    if(NOT GCOVR_EXECUTABLE)
        message(FATAL_ERROR "ENABLE_COVERAGE is ON, but gcovr was not found")
    endif()

    set(GCOVR_GCOV_EXECUTABLE gcov)
    if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
        find_program(LLVM_COV_EXECUTABLE NAMES llvm-cov-20 llvm-cov)
        if(NOT LLVM_COV_EXECUTABLE)
            message(FATAL_ERROR "ENABLE_COVERAGE with Clang requires llvm-cov")
        endif()
        set(GCOVR_GCOV_EXECUTABLE "${LLVM_COV_EXECUTABLE} gcov")
    endif()

    add_compile_options(--coverage -O0 -g)
    add_link_options(--coverage)
endif()

if(ENABLE_PROTOBUF)
    find_package(Protobuf REQUIRED)
    add_subdirectory(proto)

    list(APPEND PROTO_SRCS
        src/serialization/ModelSerialization.cpp
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

if(CPPCHECK_EXECUTABLE)
    add_custom_target(cppcheck
        COMMAND ${CPPCHECK_EXECUTABLE}
            ${CPPCHECK_ARGS}
            -I ${CMAKE_CURRENT_SOURCE_DIR}/include
            ${CMAKE_CURRENT_SOURCE_DIR}/main.cpp
            ${CPPCHECK_SOURCES}
            ${HEADERS}
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        COMMENT "Running cppcheck"
        VERBATIM
    )
endif()

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
