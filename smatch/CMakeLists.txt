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

message(@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@)
message(${OpenCV_LIBS} )
message(@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@)

# #BOOST
# find_package(Boost REQUIRED COMPONENTS filesystem system program_options thread iostreams graph)
# if(Boost_FOUND)
#   include_directories(${Boost_INCLUDE_DIR}) # cached
# endif(Boost_FOUND)


## Specify additional locations of header files
## Your package locations should be listed before other locations
include_directories(
  include/
  include/smatch
)


add_library(smatch STATIC
    src/smatch/census.cpp
    src/smatch/DSI.cpp
    src/smatch/postprocess.cpp
    src/smatch/sgm.cpp
)

#################
## WebService ###
#################

add_executable(example src/run.cpp)
target_link_libraries(example smatch ${OpenCV_LIBS} )



# add_executable(${PROJECT_NAME}_client_stub src/nodes/testing/atlas_client_stub.cpp)
# target_link_libraries(${PROJECT_NAME}_client_stub  ${Boost_LIBRARIES} proto)
