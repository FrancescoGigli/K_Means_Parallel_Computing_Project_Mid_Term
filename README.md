
# K-Means Parallel Computing Project

This repository contains the implementation of the K-Means clustering algorithm using three different approaches: sequential, parallel (OpenMP), and GPU-accelerated (CUDA). The project is designed as a mid-term evaluation for a parallel computing course, and it demonstrates how the performance of the K-Means algorithm can be optimized by leveraging parallel computing techniques.

## Project Overview

### Sequential Version
The sequential version (`sequential_main.cpp`) serves as the baseline implementation of the K-Means algorithm. It assigns points to clusters and updates centroids in a loop, running for a fixed number of 20 iterations. Results for different configurations of points and clusters are saved to a results file.

### Parallel Version (OpenMP)
The parallel version (`parallel_main.cpp`) utilizes OpenMP to speed up the process of point assignment and centroid updates. The number of threads can be configured, and the program outputs the total execution time and per-iteration time for each run. Special tests are conducted to compare the performance across different thread counts.

### CUDA Version
The CUDA version (`cuda_main.cu`) accelerates the K-Means algorithm by offloading the computation to the GPU. The code is designed to take advantage of modern CUDA architectures and can handle large datasets efficiently.

## File Structure

- **`Point.h`**: Defines the `Point` class, which represents individual points with X and Y coordinates, and includes methods for managing cluster assignment.
- **`Cluster.h`**: Defines the `Cluster` class, which represents cluster centroids and provides methods for updating the centroid based on assigned points.
- **`common_functions.h`**: Contains utility functions shared across all implementations, including point generation, cluster initialization, and distance calculation.
- **`sequential_main.cpp`**: The main implementation of the sequential K-Means algorithm.
- **`parallel_main.cpp`**: The main implementation of the OpenMP-parallelized K-Means algorithm.
- **`cuda_main.cu`**: The CUDA-accelerated implementation of the K-Means algorithm.
- **`CMakeLists.txt`**: Configuration file for building the project with CMake, supporting CUDA and OpenMP.

## Building the Project

To build the project, ensure that you have CUDA and OpenMP installed, along with a modern C++ compiler.

1. Clone the repository:
   ```bash
   git clone https://github.com/your-repo/kmeans_parallel_computing.git
   cd kmeans_parallel_computing
   ```

2. Run the desired executable:
   - For the sequential version:
     ```bash
     ./kmeans_sequential
     ```
   - For the OpenMP parallel version:
     ```bash
     ./kmeans_openmp
     ```
   - For the CUDA version:
     ```bash
     ./kmeans_cuda
     ```

## Performance Comparison

The project includes a comparison between the sequential and parallel versions by running tests on datasets with varying numbers of points and clusters. The results include the total execution time and the time per iteration for each configuration.

### Special Runs
- Sequential and OpenMP: Tests are conducted on datasets with up to 1 million points and up to 1,000 clusters.
- OpenMP: The number of threads is varied from 2 to 16 to evaluate the impact of thread count on performance.

## License

This project is licensed under the MIT License. See the LICENSE file for details.
