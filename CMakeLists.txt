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

add_executable(Network
    main.cpp
    Hopfield.cpp
    Model.cpp
    HopfieldLayer.cpp
    Layer.cpp
    LearningRule.cpp
    Perceptron.cpp
    PerceptronLayer.cpp
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

target_include_directories(Network PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(Network PRIVATE ${LIBRARIES})