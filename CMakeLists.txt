cmake_minimum_required(VERSION 3.15)
project(VisualComputing2025 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCV REQUIRED)

add_executable(vc_panorama src/panorama.cpp)
target_link_libraries(vc_panorama PRIVATE ${OpenCV_LIBS})
