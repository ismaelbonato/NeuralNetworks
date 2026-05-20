find_package(Catch2 3 REQUIRED)

set(TEST_EXECUTABLE "${PROJECT_NAME}-tests")

add_executable(${TEST_EXECUTABLE}
    ActivationTests.cpp
    ConvolutionGeometryTests.cpp
    ConvolutionalLayerTests.cpp
    FeedForwardTests.cpp
    HopfieldTests.cpp
    LayerTests.cpp
    PerceptronTests.cpp
    SmokeTests.cpp
    TensorTests.cpp
    SerializationTests.cpp
)

target_link_libraries(${TEST_EXECUTABLE} PRIVATE
    ${PROJECT_NAME}
    Catch2::Catch2WithMain
)

network_enable_warnings(${TEST_EXECUTABLE})

include(Catch)
catch_discover_tests(${TEST_EXECUTABLE})

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
        DEPENDS ${TEST_EXECUTABLE}
        COMMENT "Running tests and generating coverage report"
        VERBATIM
    )
endif()
