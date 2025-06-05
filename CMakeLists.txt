cmake_minimum_required(VERSION 3.10)
project(NetworkFramework)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED)

set(LIBRARIES
    ${OpenCV_LIBS}
    Eigen3::Eigen
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

target_include_directories(Network PRIVATE )
target_include_directories(Network PRIVATE ${OpenCV_INCLUDE_DIRS} ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(Network PRIVATE ${LIBRARIES})