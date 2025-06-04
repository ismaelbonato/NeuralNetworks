cmake_minimum_required(VERSION 3.10)
project(HopfieldNetwork)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED)

set(LIBRARIES
    ${OpenCV_LIBS}
    Eigen3::Eigen
)

add_executable(hopfield
    main.cpp
    Hopfield.cpp
    Model.cpp
    HopfieldLayer.cpp
    Layer.cpp
    LearningRule.cpp
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
target_link_libraries(hopfield PRIVATE ${LIBRARIES})