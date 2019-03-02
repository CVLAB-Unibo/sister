cmake_minimum_required(VERSION 2.8)

project(TestKinect)

set (CMAKE_CXX_STANDARD 11)

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

#Open-CV
find_package(OpenCV 4.0 REQUIRED )
if(OpenCV_INCLUDE_DIRS)
	include_directories(${OpenCV_INCLUDE_DIRS})
	link_directories(${OpenCV_LIB_DIRS})
endif(OpenCV_INCLUDE_DIRS)

#freenect 2
find_package(freenect2)
include_directories(${freenect2_INCLUDE_DIR})
link_directories(${freenect2_LIBRARY_DIRS})

add_executable(kinect ${PROJECT_SOURCE_DIR}/src/main.cpp)
target_link_libraries(kinect ${OpenCV_LIBS} ${freenect2_LIBRARY})