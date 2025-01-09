#include "common_functions.h"
#include "Cluster.h"
#include "Point.h"
#include <iostream>
#include <fstream>
#include <chrono>
#include <vector>

// Macro to check for CUDA errors
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error in %s (%s:%d): %s\n", \
                    #call, __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

// Kernel to assign points to the nearest cluster on the GPU
__global__ void assign_points(Point* points, Cluster* clusters, int num_points, int num_clusters) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < num_points) {
        float min_dist = squared_euclidean_distance(points[i], clusters[0]);
        int nearest_cluster_id = 0;

        for (int j = 1; j < num_clusters; ++j) {
            float dist = squared_euclidean_distance(points[i], clusters[j]);
            if (dist < min_dist) {
                min_dist = dist;
                nearest_cluster_id = j;
            }
        }
        points[i].set_id(nearest_cluster_id);
    }
}

// Kernel to accumulate point coordinates for each cluster
__global__ void accumulate_points(Point* points, Cluster* clusters, int num_points) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < num_points) {
        int cluster_id = points[i].get_cluster_id();

        // Use atomic operations to safely accumulate values
        clusters[cluster_id].atomic_add_to_new_x_coord(points[i].get_x());
        clusters[cluster_id].atomic_add_to_new_y_coord(points[i].get_y());
        clusters[cluster_id].atomic_increment_size();
    }
}

// Kernel to update cluster centroids on the GPU
__global__ void update_centroids(Cluster* clusters, int num_clusters) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < num_clusters && clusters[i].get_size() > 0) {
        float new_x = clusters[i].get_new_x_coord() / static_cast<float>(clusters[i].get_size());
        float new_y = clusters[i].get_new_y_coord() / static_cast<float>(clusters[i].get_size());
        clusters[i].set_x_coord(new_x);
        clusters[i].set_y_coord(new_y);
        clusters[i].reset_values(); // Reset for the next iteration
    }
}

// K-means algorithm on the GPU, performing 20 iterations
void kmeans_cuda(std::vector<Point>& points, std::vector<Cluster>& clusters, double& total_time, int blockSize) {
    const int max_iterations = 20;
    int iterations = 0;

    // Allocate memory on the GPU
    Point* d_points;
    Cluster* d_clusters;

    CUDA_CHECK(cudaMalloc(&d_points, points.size() * sizeof(Point)));
    CUDA_CHECK(cudaMalloc(&d_clusters, clusters.size() * sizeof(Cluster)));

    // Copy data to the GPU
    CUDA_CHECK(cudaMemcpy(d_points, points.data(), points.size() * sizeof(Point), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_clusters, clusters.data(), clusters.size() * sizeof(Cluster), cudaMemcpyHostToDevice));

    // Configure grid and block dimensions
    int numBlocksPoints = (points.size() + blockSize - 1) / blockSize;
    int numBlocksClusters = (clusters.size() + blockSize - 1) / blockSize;

    while (iterations < max_iterations) {
        auto start_iter = std::chrono::high_resolution_clock::now();

        // Step 1: Assign points to the nearest cluster (GPU)
        assign_points<<<numBlocksPoints, blockSize>>>(d_points, d_clusters, points.size(), clusters.size());
        CUDA_CHECK(cudaGetLastError());

        // Step 2: Accumulate point coordinates for each cluster (GPU)
        accumulate_points<<<numBlocksPoints, blockSize>>>(d_points, d_clusters, points.size());
        CUDA_CHECK(cudaGetLastError());

        // Step 3: Update cluster centroids (GPU)
        update_centroids<<<numBlocksClusters, blockSize>>>(d_clusters, clusters.size());
        CUDA_CHECK(cudaGetLastError());

        // Synchronize after each iteration
        CUDA_CHECK(cudaDeviceSynchronize());

        auto end_iter = std::chrono::high_resolution_clock::now();
        double iteration_time = std::chrono::duration<double>(end_iter - start_iter).count();
        total_time += iteration_time; // Add to total time

        iterations++;
    }

    // Copy updated clusters back to the CPU
    CUDA_CHECK(cudaMemcpy(clusters.data(), d_clusters, clusters.size() * sizeof(Cluster), cudaMemcpyDeviceToHost));

    // Free GPU memory
    CUDA_CHECK(cudaFree(d_points));
    CUDA_CHECK(cudaFree(d_clusters));

    std::cout << "Completed " << max_iterations << " iterations on the GPU with block size " << blockSize << ".\n";
}

// Run the K-means algorithm with CUDA and save results to a file
void run_kmeans_cuda(int num_points, int num_clusters, const std::vector<int>& block_sizes, std::ofstream& output_file) {
    std::cout << "Running K-means for " << num_points << " points and " << num_clusters
              << " clusters with varying block sizes.\n";

    // Generate random points
    std::vector<Point> points = generate_points(num_points);

    // Test for each block size
    for (int block_size : block_sizes) {
        // Initialize clusters
        std::vector<Cluster> clusters = initialize_clusters(num_clusters, points);

        double total_time = 0.0;

        // Run K-means with the current block size
        kmeans_cuda(points, clusters, total_time, block_size);

        double time_per_iteration = total_time / 20.0; // 20 iterations

        // Save results to the output file
        output_file << "Configuration: " << num_points << " points, " << num_clusters
                    << " clusters, Block size: " << block_size << " (CUDA)\n";
        output_file << "Total execution time: " << total_time << " seconds\n";
        output_file << "Time per iteration: " << time_per_iteration << " seconds\n";
        output_file << "----------------------------------------\n";

        std::cout << "Completed 20 iterations on the GPU with block size " << block_size << ".\n";
    }

    std::cout << "Completed all block size configurations for " << num_points << " points and " << num_clusters << " clusters.\n";
}

int main() {
    // Open the file to save results
    std::ofstream output_file("cuda_main_results.txt");
    if (!output_file.is_open()) {
        std::cerr << "Error: Could not open the file to save results.\n";
        return EXIT_FAILURE;
    }

    // Define configurations for normal runs
    std::vector<int> num_points_list = {100000, 250000, 500000, 1000000};
    std::vector<int> num_clusters_list = {5, 10, 20};

    // Define block sizes to test
    std::vector<int> block_sizes = {32, 64, 128, 256, 512, 1024};

    // Run the algorithm for each configuration
    for (int num_points : num_points_list) {
        for (int num_clusters : num_clusters_list) {
            run_kmeans_cuda(num_points, num_clusters, block_sizes, output_file);
        }
    }

    // Special run for 1 million points with larger cluster counts
    std::vector<int> large_cluster_list = {100, 250, 500, 1000};
    output_file << "Running special CUDA K-means test for 1 million points with larger cluster counts.\n";
    std::cout << "Running special CUDA K-means test for 1 million points with larger cluster counts.\n";
    for (int num_clusters : large_cluster_list) {
        run_kmeans_cuda(1000000, num_clusters, block_sizes, output_file);
    }

    // Close the output file
    output_file.close();

    std::cout << "K-means CUDA tests completed. Results saved to 'cuda_main_results.txt'.\n";

    return EXIT_SUCCESS;
}
