cmake_minimum_required(VERSION 3.23)
project(sift)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

set(CMAKE_CXX_STANDARD 11)

add_executable(sift src/main.cpp include/pre.h src/pre.cpp)
target_link_libraries(sift ${OpenCV_LIBS})


