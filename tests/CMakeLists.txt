find_package(Catch2 3 REQUIRED)

set(TEST_EXECUTABLE "${PROJECT_NAME}-tests")

add_executable(${TEST_EXECUTABLE}
    ActivationTests.cpp
    ConvolutionalLayerTests.cpp
    FeedForwardTests.cpp
    HopfieldTests.cpp
    LayerTests.cpp
    PerceptronTests.cpp
    SmokeTests.cpp
    TensorTests.cpp
)

if(ENABLE_PROTOBUF)
    target_sources(${TEST_EXECUTABLE} PRIVATE
        SerializationTests.cpp
    )

    target_compile_definitions(${TEST_EXECUTABLE} PRIVATE
        NN_RUNTIME_TEST_OUTPUT_DIR="${CMAKE_CURRENT_BINARY_DIR}"
    )

    set_source_files_properties(SerializationTests.cpp PROPERTIES
        SKIP_LINTING TRUE
    )
endif()

target_link_libraries(${TEST_EXECUTABLE} PRIVATE
    ${PROJECT_NAME}
    Catch2::Catch2WithMain
)

if(ENABLE_PROTOBUF)
    target_link_libraries(${TEST_EXECUTABLE} PRIVATE
        nn-runtime-protobuf
    )
endif()

network_enable_warnings(${TEST_EXECUTABLE})

include(Catch)
catch_discover_tests(${TEST_EXECUTABLE})

network_add_coverage_target(coverage ${TEST_EXECUTABLE})
