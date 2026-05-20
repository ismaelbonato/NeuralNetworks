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

network_add_coverage_target(coverage ${TEST_EXECUTABLE})
