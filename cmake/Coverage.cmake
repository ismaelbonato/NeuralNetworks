option(ENABLE_COVERAGE "Build with coverage instrumentation and add a coverage target" OFF)

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

function(network_add_coverage_target target_name test_dependency)
    if(ENABLE_COVERAGE)
        add_custom_target(${target_name}
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
            DEPENDS ${test_dependency}
            COMMENT "Running tests and generating coverage report"
            VERBATIM
        )
    endif()
endfunction()
