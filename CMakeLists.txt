cmake_minimum_required(VERSION 4.0)

#set(CMAKE_C_COMPILER "clang-19")
#set(CMAKE_CXX_COMPILER "clang++-19")

project(NetworkFramework)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Debug CACHE STRING "Build type" FORCE)
endif()

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)


find_package(OpenCV REQUIRED)
#find_package(OpenMP REQUIRED REQUIRED)

#if(NOT OpenMP_CXX_FOUND)
#    message(FATAL_ERROR "OpenMP C++ support not found. Please ensure OpenMP is installed and configured correctly.")
#endif()

set(LIBRARIES
    ${OpenCV_LIBS}
 #   OpenMP::OpenMP_CXX
)

file(GLOB_RECURSE SOURCES "src/*.cpp")
file(GLOB_RECURSE HEADERS "include/*.cpp")


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



#target_link_libraries(NetworkLib PUBLIC ${LIBRARIES})