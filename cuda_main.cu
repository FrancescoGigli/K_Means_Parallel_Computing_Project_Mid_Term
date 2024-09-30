#include "common_functions.h"  // Include common functions
#include "Cluster.h"            // Include the Cluster class
#include <iostream>
#include <fstream>
#include <chrono>               // For time measurement

// Device function to calculate squared Euclidean distance on the GPU
__device__ double squared_euclidean_distance_device(const Point& pt, const Cluster& cl) {
    return pow(pt.get_x() - cl.get_x(), 2) + pow(pt.get_y() - cl.get_y(), 2);
}

// Kernel function to assign points to the nearest cluster (on the GPU)
__global__ void assign_points(Point* points, Cluster* clusters, int num_points, int num_clusters) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < num_points) {
        double min_dist = squared_euclidean_distance_device(points[i], clusters[0]);
        int nearest_cluster_id = 0;

        for (int j = 1; j < num_clusters; ++j) {
            double dist = squared_euclidean_distance_device(points[i], clusters[j]);
            if (dist < min_dist) {
                min_dist = dist;
                nearest_cluster_id = j;
            }
        }
        points[i].set_id(nearest_cluster_id);
    }
}

// K-means algorithm with CUDA, running 20 iterations
void kmeans_cuda(std::vector<Point>& points, std::vector<Cluster>& clusters, double& total_time) {
    const int max_iterations = 20;
    int iterations = 0;

    // Allocate memory on the GPU
    Point* d_points;
    Cluster* d_clusters;
    cudaMalloc(&d_points, points.size() * sizeof(Point));
    cudaMalloc(&d_clusters, clusters.size() * sizeof(Cluster));

    // Copy data to the GPU
    cudaMemcpy(d_points, points.data(), points.size() * sizeof(Point), cudaMemcpyHostToDevice);
    cudaMemcpy(d_clusters, clusters.data(), clusters.size() * sizeof(Cluster), cudaMemcpyHostToDevice);

    // Set up CUDA block and grid sizes
    int blockSize = 256;
    int numBlocks = (points.size() + blockSize - 1) / blockSize;

    while (iterations < max_iterations) {
        auto start_iter = std::chrono::high_resolution_clock::now();

        // Step 1: Assign points to the nearest cluster (on GPU)
        assign_points<<<numBlocks, blockSize>>>(d_points, d_clusters, points.size(), clusters.size());

        // Copy updated points back to host (CPU)
        cudaMemcpy(points.data(), d_points, points.size() * sizeof(Point), cudaMemcpyDeviceToHost);

        // Step 2: Update cluster centroids (sequential step on CPU)
        for (auto& cluster : clusters) {
            cluster.delete_values(); // Reset values
        }

        for (const auto& point : points) {
            clusters[point.get_cluster_id()].add_point(point);
        }

        for (auto& cluster : clusters) {
            cluster.update_values();
        }

        // Copy updated clusters back to GPU
        cudaMemcpy(d_clusters, clusters.data(), clusters.size() * sizeof(Cluster), cudaMemcpyHostToDevice);

        auto end_iter = std::chrono::high_resolution_clock::now();
        double iteration_time = std::chrono::duration<double>(end_iter - start_iter).count();  // Time in seconds
        total_time += iteration_time;  // Add to total time

        iterations++;
    }

    // Free GPU memory
    cudaFree(d_points);
    cudaFree(d_clusters);

    std::cout << "Completed " << max_iterations << " iterations on the GPU.\n";
}

// Function to run the K-means algorithm with CUDA and save results to a file
void run_kmeans_cuda(int num_points, int num_clusters, std::ofstream& output_file) {
    std::cout << "Running K-means with " << num_points << " points and " << num_clusters << " clusters on the GPU.\n";

    // Generate points and initialize clusters
    std::vector<Point> points = generate_points(num_points);
    std::vector<Cluster> clusters = initialize_clusters(num_clusters, points);

    // Measure total execution time
    double total_time = 0.0;

    kmeans_cuda(points, clusters, total_time);

    // Calculate time per iteration
    double time_per_iteration = total_time / 20.0;  // We run exactly 20 iterations

    // Save the result to the output file
    output_file << "Configuration: " << num_points << " points, " << num_clusters << " clusters (CUDA)\n";
    output_file << "Total execution time: " << total_time << " seconds\n";
    output_file << "Time per iteration: " << time_per_iteration << " seconds\n";
    output_file << "----------------------------------------\n";
}

int main() {
    // Open file to save results
    std::ofstream output_file("cuda_main_results.txt");
    if (!output_file.is_open()) {
        std::cerr << "Error: Could not open the file to save results.\n";
        return 1;
    }

    // Define the different configurations for normal runs
    std::vector<int> num_points_list = {100000, 250000, 500000, 1000000};
    std::vector<int> num_clusters_list = {5, 10, 20};

    // Run the algorithm for each configuration and save results
    for (int num_points : num_points_list) {
        for (int num_clusters : num_clusters_list) {
            run_kmeans_cuda(num_points, num_clusters, output_file);
        }
    }

    // Special run for 1 million points with larger cluster numbers
    std::vector<int> large_cluster_list = {100, 250, 500, 1000};
    output_file << "Running special CUDA K-means test for 1 million points with larger cluster counts.\n";
    for (int num_clusters : large_cluster_list) {
        run_kmeans_cuda(1000000, num_clusters, output_file);
    }

    // Close the file
    output_file.close();

    return 0;
}
