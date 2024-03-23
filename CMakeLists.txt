cmake_minimum_required(VERSION 3.10)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_PREFIX_PATH /opt/intel/openvino_2023/runtime/cmake/)
set(CMAKE_BUILD_TYPE "Debug")
project(yolov8_pose_openvino)

find_package(OpenCV REQUIRED)
find_package(OpenVINO REQUIRED)

include_directories(
	./src/
	./include/)

include_directories(${OpenCV_INCLUDE_DIRS})

FILE(GLOB_RECURSE TEST_SRC
	src/*.cpp
	)

add_executable(target src/main.cpp ${TEST_SRC})

target_link_libraries(target  ${OpenCV_LIBS})
target_link_libraries(target openvino::runtime)