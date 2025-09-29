cmake_minimum_required(VERSION 3.0)
project(LibCintoRoootGeometryCXX)

find_package(ROOT REQUIRED COMPONENTS Geom)
include(${ROOT_USE_FILE})

add_library(cinto_root_geometry SHARED root_geometry.cc)
target_link_libraries(cinto_root_geometry ${ROOT_LIBRARIES})

install(TARGETS cinto_root_geometry DESTINATION "$ENV{OUT_DIR}")
