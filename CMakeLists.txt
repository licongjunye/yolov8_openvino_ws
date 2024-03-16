cmake_minimum_required(VERSION 3.10)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_PREFIX_PATH /opt/intel/openvino_2023/runtime/cmake/)
set(CMAKE_BUILD_TYPE "Debug")
project(yolov8_pose_openvino)

# 设置编译模式
#set( CMAKE_BUILD_TYPE "Release" )

#添加OPENCV库
#指定OpenCV版本，代码如下
#find_package(OpenCV 4.8 REQUIRED)
#如果不需要指定OpenCV版本，代码如下
find_package(OpenCV REQUIRED)
find_package(OpenVINO REQUIRED)

include_directories(
	./src/)


#添加OpenCV头文件
include_directories(${OpenCV_INCLUDE_DIRS})

#显示OpenCV_INCLUDE_DIRS的值
# message(${OpenCV_INCLUDE_DIRS})

FILE(GLOB_RECURSE TEST_SRC
	src/*.cpp
	)

# 添加一个可执行程序
# 语法：add_executable( 程序名 源代码文件 ）
add_executable(target main.cpp ${TEST_SRC})

# 将库文件链接到可执行程序上
target_link_libraries(target  ${OpenCV_LIBS})
target_link_libraries(target openvino::runtime)

