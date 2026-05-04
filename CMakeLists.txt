cmake_minimum_required(VERSION 3.20)

project(nn-runtime)

include(CTest)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Debug CACHE STRING "Build type" FORCE)
endif()

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(ENABLE_CPPCHECK "Run cppcheck as part of C++ builds" OFF)
option(ENABLE_CLANG_TIDY "Run clang-tidy as part of C++ builds" OFF)
option(ENABLE_COVERAGE "Build with coverage instrumentation and add a coverage target" OFF)

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

set(SOURCES
    src/base/Layer.cpp
    src/base/Model.cpp
    src/layers/ConvolutionalLayer.cpp
    src/layers/DenseLayer.cpp
    src/layers/FlattenLayer.cpp
    src/layers/HopfieldLayer.cpp
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

if(CPPCHECK_EXECUTABLE)
    add_custom_target(cppcheck
        COMMAND ${CPPCHECK_EXECUTABLE}
            ${CPPCHECK_ARGS}
            -I ${CMAKE_CURRENT_SOURCE_DIR}/include
            ${CMAKE_CURRENT_SOURCE_DIR}/main.cpp
            ${SOURCES}
            ${HEADERS}
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        COMMENT "Running cppcheck"
        VERBATIM
    )
endif()

add_library(nn-runtime
    ${SOURCES}
    ${HEADERS}
)

target_include_directories(nn-runtime PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

network_enable_warnings(nn-runtime)

if(PROJECT_NAME STREQUAL CMAKE_PROJECT_NAME)
    add_executable(nn-runtime-main main.cpp)

    target_link_libraries(nn-runtime-main PRIVATE
        nn-runtime
    )

    target_include_directories(nn-runtime-main PRIVATE
        ${CMAKE_SOURCE_DIR}/include
    )

    network_enable_warnings(nn-runtime-main)

    if(BUILD_TESTING)
        find_package(Catch2 3 REQUIRED)

        add_executable(nn-runtime-tests
            tests/ActivationTests.cpp
            tests/ConvolutionGeometryTests.cpp
            tests/ConvolutionalLayerTests.cpp
            tests/FeedForwardTests.cpp
            tests/HopfieldTests.cpp
            tests/LayerTests.cpp
            tests/PerceptronTests.cpp
            tests/SmokeTests.cpp
            tests/TensorTests.cpp
        )

        target_link_libraries(nn-runtime-tests PRIVATE
            nn-runtime
            Catch2::Catch2WithMain
        )

        include(Catch)
        catch_discover_tests(nn-runtime-tests)

        if(ENABLE_COVERAGE)
            add_custom_target(coverage
                COMMAND ${CMAKE_CTEST_COMMAND}
                    --test-dir ${CMAKE_BINARY_DIR}
                    --output-on-failure
                COMMAND ${GCOVR_EXECUTABLE}
                    --root ${CMAKE_SOURCE_DIR}
                    --gcov-executable ${GCOVR_GCOV_EXECUTABLE}
                    --filter ${CMAKE_SOURCE_DIR}/include
                    --filter ${CMAKE_SOURCE_DIR}/src
                    --exclude ${CMAKE_SOURCE_DIR}/tests
                    --html-details ${CMAKE_BINARY_DIR}/coverage.html
                    --print-summary
                WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
                DEPENDS nn-runtime-tests
                COMMENT "Running tests and generating coverage report"
                VERBATIM
            )
        endif()
    endif()
endif()
