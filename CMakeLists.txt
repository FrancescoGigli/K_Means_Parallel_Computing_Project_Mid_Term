cmake_minimum_required(VERSION 3.24)
project(kmeans LANGUAGES CXX CUDA)

# Set the C++ and CUDA standards
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_STANDARD 11)

# Set the GPU architecture
set(CMAKE_CUDA_ARCHITECTURES 75)

# Find the CUDA package
find_package(CUDA REQUIRED)

# Add executable for the CUDA version
add_executable(kmeans_cuda cuda_main.cu Point.h Cluster.h)

# Link the CUDA runtime library
target_link_libraries(kmeans_cuda ${CUDA_LIBRARIES})

# Find OpenMP package for the OpenMP version
find_package(OpenMP REQUIRED)
if(OpenMP_CXX_FOUND)
    # Add executable for the OpenMP version
    add_executable(kmeans_openmp parallel_main.cpp Point.h Cluster.h)
    # Link the OpenMP library to the executable
    target_link_libraries(kmeans_openmp OpenMP::OpenMP_CXX)
endif()

# Add executable for the sequential version
add_executable(kmeans_sequential sequential_main.cpp Point.h Cluster.h)


# Include common_functions.h in all targets
target_include_directories(kmeans_cuda PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(kmeans_openmp PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(kmeans_sequential PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

