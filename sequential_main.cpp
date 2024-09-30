#include "common_functions.h" // Include the common functions
#include "Cluster.h"           // Include the Cluster class
#include <iostream>
#include <fstream>
#include <chrono>              // For time measurement

// K-means algorithm with 20 fixed iterations
void kmeans(std::vector<Point>& points, std::vector<Cluster>& clusters, double& total_time) {
    const int max_iterations = 20;
    int iterations = 0;

    while (iterations < max_iterations) {
        auto start_iter = std::chrono::high_resolution_clock::now();

        // Step 1: Assign points to the nearest cluster
        for (auto& point : points) {
            double min_dist = squared_euclidean_distance(point, clusters[0]);
            int nearest_cluster_id = 0;

            for (int i = 1; i < clusters.size(); ++i) {
                double dist = squared_euclidean_distance(point, clusters[i]);
                if (dist < min_dist) {
                    min_dist = dist;
                    nearest_cluster_id = i;
                }
            }
            point.set_id(nearest_cluster_id);
        }

        // Step 2: Update cluster centroids
        for (auto& cluster : clusters) {
            cluster.delete_values(); // Reset values
        }

        for (const auto& point : points) {
            clusters[point.get_cluster_id()].add_point(point);
        }

        for (auto& cluster : clusters) {
            cluster.update_values();
        }

        auto end_iter = std::chrono::high_resolution_clock::now();
        double iteration_time = std::chrono::duration<double>(end_iter - start_iter).count();  // Time in seconds
        total_time += iteration_time;  // Add to total time

        iterations++;
    }

    std::cout << "Completed " << max_iterations << " iterations.\n";
}

// Function to run the K-means algorithm and save results to a file
void run_kmeans(int num_points, int num_clusters, std::ofstream& output_file) {
    std::cout << "Running K-means with " << num_points << " points and " << num_clusters << " clusters.\n";

    // Generate points and initialize clusters
    std::vector<Point> points = generate_points(num_points);
    std::vector<Cluster> clusters = initialize_clusters(num_clusters, points);

    // Measure total execution time
    double total_time = 0.0;

    kmeans(points, clusters, total_time);

    // Calculate time per iteration
    double time_per_iteration = total_time / 20.0;  // We run exactly 20 iterations

    // Save the result to the output file
    output_file << "Configuration: " << num_points << " points, " << num_clusters << " clusters\n";
    output_file << "Total execution time: " << total_time << " seconds\n";
    output_file << "Time per iteration: " << time_per_iteration << " seconds\n";
    output_file << "----------------------------------------\n";
}

int main() {
    // Open file to save results
    std::ofstream output_file("sequential_main_results.txt");
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
            run_kmeans(num_points, num_clusters, output_file);
        }
    }

    // Special run for 1 million points with larger cluster numbers
    std::vector<int> large_cluster_list = {100, 250, 500, 1000};
    output_file << "Running special K-means test for 1 million points with larger cluster counts.\n";
    for (int num_clusters : large_cluster_list) {
        run_kmeans(1000000, num_clusters, output_file);
    }

    // Close the file
    output_file.close();

    return 0;
}
