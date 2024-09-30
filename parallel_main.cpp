#include "common_functions.h"  // Include the common functions
#include "Cluster.h"            // Include the Cluster class
#include <iostream>
#include <fstream>
#include <chrono>               // For time measurement
#include <omp.h>                // OpenMP header for parallelism

// K-means algorithm with OpenMP parallelization and 20 fixed iterations
void kmeans_parallel(std::vector<Point>& points, std::vector<Cluster>& clusters, double& total_time, int num_threads) {
    const int max_iterations = 20;
    int iterations = 0;

    omp_set_num_threads(num_threads);  // Set the number of threads for parallel execution

    while (iterations < max_iterations) {
        auto start_iter = std::chrono::high_resolution_clock::now();

        // Step 1: Assign points to the nearest cluster (parallelized with OpenMP)
#pragma omp parallel for
        for (int i = 0; i < points.size(); ++i) {
            double min_dist = squared_euclidean_distance(points[i], clusters[0]);
            int nearest_cluster_id = 0;

            for (int j = 1; j < clusters.size(); ++j) {
                double dist = squared_euclidean_distance(points[i], clusters[j]);
                if (dist < min_dist) {
                    min_dist = dist;
                    nearest_cluster_id = j;
                }
            }
            points[i].set_id(nearest_cluster_id);
        }

        // Step 2: Update cluster centroids (sequential step)
        for (auto& cluster : clusters) {
            cluster.delete_values();  // Reset cluster values
        }

        // Accumulate points in clusters (parallelized with OpenMP)
#pragma omp parallel for
        for (int i = 0; i < points.size(); ++i) {
            clusters[points[i].get_cluster_id()].add_point(points[i]);
        }

        // Update centroids (sequential step)
        for (auto& cluster : clusters) {
            cluster.update_values();
        }

        auto end_iter = std::chrono::high_resolution_clock::now();
        double iteration_time = std::chrono::duration<double>(end_iter - start_iter).count();  // Time in seconds
        total_time += iteration_time;  // Add to total time

        iterations++;
    }

    std::cout << "Completed " << max_iterations << " iterations with " << num_threads << " threads.\n";
}

// Function to run the K-means algorithm with OpenMP and save results to a file
void run_kmeans_parallel(int num_points, int num_clusters, int num_threads, std::ofstream& output_file) {
    std::cout << "Running K-means with " << num_points << " points, " << num_clusters << " clusters, and " << num_threads << " threads.\n";

    // Generate points and initialize clusters
    std::vector<Point> points = generate_points(num_points);
    std::vector<Cluster> clusters = initialize_clusters(num_clusters, points);

    // Measure total execution time
    double total_time = 0.0;

    kmeans_parallel(points, clusters, total_time, num_threads);

    // Calculate time per iteration
    double time_per_iteration = total_time / 20.0;  // We run exactly 20 iterations

    // Save the result to the output file
    output_file << "Configuration: " << num_points << " points, " << num_clusters << " clusters, " << num_threads << " threads\n";
    output_file << "Total execution time: " << total_time << " seconds\n";
    output_file << "Time per iteration: " << time_per_iteration << " seconds\n";
    output_file << "----------------------------------------\n";
}

int main() {
    // Open file to save results
    std::ofstream output_file("parallel_main_openMP_results.txt");
    if (!output_file.is_open()) {
        std::cerr << "Error: Could not open the file to save results.\n";
        return 1;
    }

    // Define the different configurations for normal runs
    std::vector<int> num_points_list = {100000, 250000, 500000, 1000000};
    std::vector<int> num_clusters_list = {5, 10, 20};

    // Run the algorithm for each configuration with a default of 4 threads
    for (int num_points : num_points_list) {
        for (int num_clusters : num_clusters_list) {
            run_kmeans_parallel(num_points, num_clusters, 4, output_file);  // Using 4 threads by default
        }
    }

    // Special run for 1 million points, 20 clusters, with varying number of threads
    std::vector<int> thread_list = {2, 4, 6, 8,10,12,14,16};
    output_file << "Running special parallel K-means test for 1 million points with 20 clusters and varying thread counts.\n";
    for (int num_threads : thread_list) {
        run_kmeans_parallel(1000000, 20, num_threads, output_file);
    }

    // Close the file
    output_file.close();

    return 0;
}
