cmake_minimum_required(VERSION 3.20)


# Use Clang 20 if available
set(CMAKE_C_COMPILER "clang-20" CACHE STRING "C compiler" FORCE)
set(CMAKE_CXX_COMPILER "clang++-20" CACHE STRING "C++ compiler" FORCE)

project(NetworkFramework)

include(CTest)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Debug CACHE STRING "Build type" FORCE)
endif()

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(ENABLE_CPPCHECK "Run cppcheck as part of C++ builds" ON)

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

option(ENABLE_CLANG_TIDY "Run clang-tidy as part of C++ builds" ON)

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

find_package(OpenCV REQUIRED)
#find_package(OpenMP REQUIRED)

#if(NOT OpenMP_CXX_FOUND)
#    message(FATAL_ERROR "OpenMP C++ support not found. Please ensure OpenMP is installed and configured correctly.")
#endif()

set(LIBRARIES
    ${OpenCV_LIBS}
  #  OpenMP::OpenMP_CXX
)

file(GLOB_RECURSE SOURCES "src/*.cpp")
file(GLOB_RECURSE HEADERS "include/*.h")

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

add_library(NetworkLib
    Helper.h
    ${SOURCES}
    ${HEADERS}
)

# Set the library version
if(PROJECT_NAME STREQUAL CMAKE_PROJECT_NAME)
    # Building directly
    add_executable(Network main.cpp)
    target_link_libraries(Network PRIVATE NetworkLib ${LIBRARIES})

    target_compile_options(Network PRIVATE
    -Wall
    -Wextra
    -Wpedantic
    -Wshadow
    -Wconversion
    -Wsign-conversion
    -Wold-style-cast
    -Wnon-virtual-dtor
    )

    target_include_directories(Network PRIVATE 
        ${OpenCV_INCLUDE_DIRS} 
        ${CMAKE_SOURCE_DIR}/include
    )

    if(BUILD_TESTING)
        find_package(Catch2 3 REQUIRED)

        add_executable(NetworkTests
            tests/FeedForwardTests.cpp
            tests/SmokeTests.cpp
        )

        target_link_libraries(NetworkTests PRIVATE
            NetworkLib
            Catch2::Catch2WithMain
        )

        include(Catch)
        catch_discover_tests(NetworkTests)
    endif()

endif()


# Add extra warning flags
target_compile_options(NetworkLib PRIVATE
    -Wall
    -Wextra
    -Wpedantic
    -Wshadow
    -Wconversion
    -Wsign-conversion
    -Wold-style-cast
    -Wnon-virtual-dtor
)


target_include_directories(NetworkLib PUBLIC 
    ${OpenCV_INCLUDE_DIRS} 
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)