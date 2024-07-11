cmake_minimum_required(VERSION 3.10)
project(XFeatTensorRT)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Suppress deprecated function warnings for GCC and Clang
if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    add_compile_options(-Wno-deprecated-declarations)
endif()

# Find CUDA
find_package(CUDA REQUIRED)

#Find OpenCV
find_package(OpenCV REQUIRED)

#Find YAML-cpp
find_package(yaml-cpp REQUIRED)

#Find LibTorch
find_package(Torch REQUIRED PATHS ${PROJECT_SOURCE_DIR}/libtorch)
message(STATUS "Found Torch include is ${TORCH_INCLUDE_DIRS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

# TensorRT paths 
set(TENSORRT_ROOT ~/Downloads/TensorRT-8.6.1.6)
set(TENSORRT_INCLUDE_DIR ${TENSORRT_ROOT}/include)
set(TENSORRT_LIBRARY_DIR ${TENSORRT_ROOT}/lib)

# Include directories
include_directories(include/ ${CUDA_INCLUDE_DIRS} ${TENSORRT_INCLUDE_DIR} ${OpenCV_INCLUDE_DIRS} ${TORCH_INCLUDE_DIRS})

# Create XFeat library
add_library(xfeat_lib SHARED
    src/xfeat.cpp
    include/xfeat.h 
    include/InterpolateSparse2D.h   
)

# Link directories
link_directories(${CUDA_LIBRARY_DIRS} ${TENSORRT_LIBRARY_DIR})

# Add executable
add_executable(xample src/xample.cpp)

# Link libraries
target_link_libraries(xfeat_lib ${CUDA_LIBRARIES} ${OpenCV_LIBS} ${TORCH_LIBRARIES} nvinfer nvonnxparser nvinfer_plugin yaml-cpp)
target_link_libraries(xample xfeat_lib ${TORCH_LIBRARIES})