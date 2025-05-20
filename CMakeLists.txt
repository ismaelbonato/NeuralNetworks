cmake_minimum_required(VERSION 3.10)
project(HopfieldNetwork)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCV REQUIRED)

add_executable(hopfield
    main.cpp
    Network.cpp
    Neuron.cpp
)

# Add extra warning flags
target_compile_options(hopfield PRIVATE
    -Wall
    -Wextra
    -Wpedantic
    -Wshadow
    -Wconversion
    -Wsign-conversion
    -Wold-style-cast
    -Wnon-virtual-dtor
)

target_include_directories(hopfield PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(hopfield PRIVATE ${OpenCV_LIBS})