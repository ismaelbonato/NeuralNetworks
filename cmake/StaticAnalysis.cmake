option(ENABLE_CPPCHECK "Run cppcheck as part of C++ builds" ON)
option(ENABLE_CLANG_TIDY "Run clang-tidy as part of C++ builds" ON)

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

function(network_add_cppcheck_target target_name)
    set(options)
    set(one_value_args)
    set(multi_value_args SOURCES HEADERS)
    cmake_parse_arguments(NETWORK_CPPCHECK
        "${options}"
        "${one_value_args}"
        "${multi_value_args}"
        ${ARGN}
    )

    if(CPPCHECK_EXECUTABLE)
        add_custom_target(${target_name}
            COMMAND ${CPPCHECK_EXECUTABLE}
                ${CPPCHECK_ARGS}
                -I ${CMAKE_CURRENT_SOURCE_DIR}/include
                ${CMAKE_CURRENT_SOURCE_DIR}/main.cpp
                ${NETWORK_CPPCHECK_SOURCES}
                ${NETWORK_CPPCHECK_HEADERS}
            WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
            COMMENT "Running cppcheck"
            VERBATIM
        )
    endif()
endfunction()
