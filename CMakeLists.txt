cmake_minimum_required(VERSION 3.0)

set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14")

if(MSVC)
	add_compile_options(/W4)
else()
	add_compile_options(-Wall )
endif()

find_package( OpenCV REQUIRED)

project(computer_vision_project1)

add_executable(
	LineDetector
	main.cpp
)

target_link_libraries( LineDetector ${OpenCV_LIBS} )