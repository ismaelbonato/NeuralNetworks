cmake_minimum_required(VERSION 4.0)

set(CMAKE_C_COMPILER "clang-19")
set(CMAKE_CXX_COMPILER "clang++-19")

project(NetworkFramework)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Debug CACHE STRING "Build type" FORCE)
endif()

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)


find_package(OpenCV REQUIRED)
find_package(OpenMP REQUIRED REQUIRED)

if(NOT OpenMP_CXX_FOUND)
    message(FATAL_ERROR "OpenMP C++ support not found. Please ensure OpenMP is installed and configured correctly.")
endif()

set(LIBRARIES
    ${OpenCV_LIBS}
    OpenMP::OpenMP_CXX
)

file(GLOB_RECURSE SOURCES "src/*.cpp")
file(GLOB_RECURSE HEADERS "include/*.cpp")

add_executable(Network
    main.cpp
    Helper.h
    ${SOURCES}
    ${HEADERS}
)

# Add extra warning flags
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

target_compile_definitions(Network PRIVATE PARALLEL)

# Enable modules for GCC/Clang (optional, but recommended for portability)
#if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
#    target_compile_options(Network PRIVATE -fmodules-ts)
#endif()

target_include_directories(Network PRIVATE )
target_include_directories(Network PRIVATE ${OpenCV_INCLUDE_DIRS} ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(Network PRIVATE ${LIBRARIES})