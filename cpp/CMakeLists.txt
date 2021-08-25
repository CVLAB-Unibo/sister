cmake_minimum_required(VERSION 2.8.3)
project(atlas)

set(CMAKE_MACOSX_RPATH 1)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17 -msse4.1 -msse4.2 -O3")

# OpenMP
find_package(OpenMP)
if(OPENMP_FOUND)
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
  message (STATUS "Found OpenMP")
else(OPENMP_FOUND)
  message (STATUS "Not found OpenMP")
endif(OPENMP_FOUND)

#Eigen3
IF (WIN32)
  set(EIGEN_DIR "" CACHE PATH  "Eigen3 install folder")
  include_directories( ${EIGEN_DIR} )
ELSE()
  find_package(Eigen3 REQUIRED)
  include_directories( ${EIGEN3_INCLUDE_DIR} )
ENDIF()

#OpenCV
find_package(OpenCV REQUIRED )
if(OpenCV_INCLUDE_DIRS)
  include_directories(${OpenCV_INCLUDE_DIRS})
endif(OpenCV_INCLUDE_DIRS)


#####################
## Sister Library ###
#####################

include_directories(
  include/
  include/sister
)

add_library(sister STATIC
    src/sister/census.cpp
    src/sister/DSI.cpp
    src/sister/postprocess.cpp
    src/sister/sgm.cpp
)

##################################
## Compute Disparities example ###
##################################

add_executable(compute_disp src/compute_disp.cpp)
target_link_libraries(compute_disp sister ${OpenCV_LIBS} )

