cmake_minimum_required(VERSION 2.8.8 FATAL_ERROR)


project(Imgui)

SET(LIBRARY_OUTPUT_PATH ${PROJECT_BINARY_DIR}/libs)


############
#group files
############
file(GLOB_RECURSE IMGUI_CURRENT_HEADERS  *.h *.hpp)
file(GLOB IMGUI_CURRENT_SOURCES  *.cpp)
source_group("Include" FILES ${IMGUI_CURRENT_HEADERS}) 
source_group("Source" FILES ${IMGUI_CURRENT_SOURCES})


############
#include dir
############
set(Imgui_INCLUDE_DIR CMAKE_CURRENT_SOURCE_DIR)
INCLUDE_DIRECTORIES(Imgui_INCLUDE_DIR)


############
#  feed back
############
mark_as_advanced(Imgui_INCLUDE_DIR)


ADD_LIBRARY(Imgui STATIC ${IMGUI_CURRENT_HEADERS} ${IMGUI_CURRENT_SOURCES})