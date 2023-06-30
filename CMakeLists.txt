cmake_minimum_required(VERSION 3.9 FATAL_ERROR)

project(MRPNN LANGUAGES CXX CUDA)

enable_language(CUDA)

IF (WIN32)

MESSAGE(STATUS "Is windows")
SET(CMAKE_CXX_FLAGS "/openmp /Ot")

ELSE ()

MESSAGE(STATUS "Is linux")
add_definitions(-DLINUX)

ENDIF ()

Option(GUI OFF)
Option(RTX30XX OFF)
Option(TENSORCORE OFF)
Option(CompileRPNN OFF)
Option(DISABLE_FAST_MATH OFF)

if (GUI)
    add_definitions(-DGUI)
    find_package(GLEW)
    find_package(glfw3)
    add_subdirectory(Imgui)
	INCLUDE_DIRECTORIES(Imgui_INCLUDE_DIR)
endif()

if (CompileRPNN)
    add_definitions(-DCRPNN)
endif()

if (TENSORCORE)
    add_definitions(-DENBALE_TENSOR)
endif()


file(GLOB CORE_HEADERS core/*.h *core/*.hpp core/*.cuh)
file(GLOB CORE_SOURCES core/*.cpp core/*.cu)

source_group("Core Include" FILES ${CORE_HEADERS}) 
source_group("Core Source" FILES ${CORE_SOURCES})

include_directories(${PROJECT_SOURCE_DIR}/core)
  
add_executable(DataGen ${CORE_HEADERS} ${CORE_SOURCES} dataGen.cu)
set_target_properties(DataGen PROPERTIES 
CUDA_SEPARABLE_COMPILATION ON
CXX_STANDARD_REQUIRED ON
CXX_STANDARD 17)
target_compile_definitions(DataGen PRIVATE _HAS_STD_BYTE=0)

add_executable(VolumeRender ${CORE_HEADERS} ${CORE_SOURCES} main.cu)
set_target_properties(VolumeRender PROPERTIES
CUDA_SEPARABLE_COMPILATION ON
CXX_STANDARD_REQUIRED ON
CXX_STANDARD 17)
target_compile_definitions(VolumeRender PRIVATE _HAS_STD_BYTE=0)

add_executable(Test ${CORE_HEADERS} ${CORE_SOURCES} test.cu)
set_target_properties(Test PROPERTIES
CUDA_SEPARABLE_COMPILATION ON
CXX_STANDARD_REQUIRED ON
CXX_STANDARD 17)
target_compile_definitions(Test PRIVATE _HAS_STD_BYTE=0)


if (DISABLE_FAST_MATH)
    if (RTX30XX)
    target_compile_options(VolumeRender PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-arch=sm_86 -maxrregcount=168>)
    target_compile_options(Test PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-arch=sm_86 -maxrregcount=168>)
    target_compile_options(DataGen PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-arch=sm_86 -maxrregcount=168>)
    else()
    target_compile_options(VolumeRender PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-arch=sm_75 -maxrregcount=168>)
    target_compile_options(Test PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-arch=sm_75 -maxrregcount=168>)
    target_compile_options(DataGen PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-arch=sm_75 -maxrregcount=168>)
    endif()
else()
    if (RTX30XX)
    target_compile_options(VolumeRender PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-use_fast_math -arch=sm_86 -maxrregcount=168>)
    target_compile_options(Test PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-use_fast_math -arch=sm_86 -maxrregcount=168>)
    target_compile_options(DataGen PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-use_fast_math -arch=sm_86 -maxrregcount=168>)
    else()
    target_compile_options(VolumeRender PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-use_fast_math -arch=sm_75 -maxrregcount=168>)
    target_compile_options(Test PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-use_fast_math -arch=sm_75 -maxrregcount=168>)
    target_compile_options(DataGen PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-use_fast_math -arch=sm_75 -maxrregcount=168>)
    endif()
endif()

if (GUI)
    target_link_libraries(VolumeRender GLEW::GLEW glfw Imgui) 
    add_dependencies(VolumeRender Imgui)
    target_link_libraries(Test GLEW::GLEW glfw Imgui) 
    add_dependencies(Test Imgui)
    target_link_libraries(DataGen GLEW::GLEW glfw Imgui) 
    add_dependencies(DataGen Imgui)
endif()