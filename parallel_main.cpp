// parallel_main.cpp
#include "common_functions.h"
#include "Cluster.h"
#include <iostream>
#include <fstream>
#include <chrono>
#include <omp.h>

// Enumeration for synchronization methods
enum SyncMethod {
    CRITICAL,
    ATOMIC
};

// Function to assign points to the nearest cluster
void assign_points_to_clusters(std::vector<Point>& points, const std::vector<Cluster>& clusters) {
#pragma omp parallel for schedule(static)
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
}

// Function to accumulate points into clusters using critical sections
void accumulate_critical(const std::vector<Point>& points, std::vector<Cluster>& clusters) {
#pragma omp parallel for schedule(static)
    for (int i = 0; i < points.size(); ++i) {
        int cluster_id = points[i].get_cluster_id();
#pragma omp critical
        {
            clusters[cluster_id].add_point(points[i]);
        }
    }
}

// Function to accumulate points into clusters using atomic operations
void accumulate_atomic(const std::vector<Point>& points, std::vector<Cluster>& clusters) {
#pragma omp parallel for schedule(static)
    for (int i = 0; i < points.size(); ++i) {
        int cluster_id = points[i].get_cluster_id();
        //add_point_atomic have atomic operation
        clusters[cluster_id].add_point_atomic(points[i]);
    }
}

// Generic K-means function that accepts the synchronization method
void kmeans_parallel(std::vector<Point>& points, std::vector<Cluster>& clusters, double& total_time, int num_threads, SyncMethod method) {
    const int max_iterations = 20;
    int iterations = 0;

    omp_set_num_threads(num_threads); // Set the number of threads for parallel execution

    // Select the accumulation function based on the synchronization method
    void (*accumulate)(const std::vector<Point>&, std::vector<Cluster>&) = nullptr;
    if (method == CRITICAL) {
        accumulate = accumulate_critical;
    } else if (method == ATOMIC) {
        accumulate = accumulate_atomic;
    } else {
        std::cerr << "Unsupported synchronization method.\n";
        return;
    }

    while (iterations < max_iterations) {
        auto start_iter = std::chrono::high_resolution_clock::now();

        // Step 1: Assign points to the nearest cluster
        assign_points_to_clusters(points, clusters);

        // Step 2: Reset cluster values in parallel
#pragma omp parallel for schedule(static)
        for (int i = 0; i < clusters.size(); ++i) {
            clusters[i].reset_values();
        }

        // Step 3: Accumulate points into clusters using the selected method
        accumulate(points, clusters);

        // Step 4: Update centroids in parallel
#pragma omp parallel for schedule(static)
        for (int i = 0; i < clusters.size(); ++i) {
            clusters[i].update_centroid();
        }

        auto end_iter = std::chrono::high_resolution_clock::now();
        double iteration_time = std::chrono::duration<double>(end_iter - start_iter).count(); // Time in seconds
        total_time += iteration_time; // Add to total time

        iterations++;
    }

    std::string method_str = (method == CRITICAL) ? "critical" : "atomic";
    std::cout << "Completed " << max_iterations << " iterations with " << num_threads << " threads (" << method_str << ").\n";
}

// Function to run both synchronization methods and save the results to a file
void run_kmeans_with_both_methods(int num_points, int num_clusters, int num_threads, std::ofstream& output_file) {
    std::cout << "Running K-means with " << num_points << " points, " << num_clusters << " clusters, and " << num_threads << " threads.\n";

    // Generate points and initialize clusters
    std::vector<Point> points = generate_points(num_points);
    std::vector<Cluster> clusters = initialize_clusters(num_clusters, points);

    // Run K-means with critical sections
    double total_time_critical = 0.0;
    kmeans_parallel(points, clusters, total_time_critical, num_threads, CRITICAL);

    // Reinitialize clusters for atomic operations
    clusters = initialize_clusters(num_clusters, points);

    // Run K-means with atomic operations
    double total_time_atomic = 0.0;
    kmeans_parallel(points, clusters, total_time_atomic, num_threads, ATOMIC);

    // Calculate time per iteration
    double time_per_iteration_critical = total_time_critical / 20.0;
    double time_per_iteration_atomic = total_time_atomic / 20.0;

    // Save the results to the output file
    output_file << "Configuration: " << num_points << " points, " << num_clusters << " clusters, " << num_threads << " threads\n";
    output_file << "Total execution time (critical): " << total_time_critical << " seconds\n";
    output_file << "Time per iteration (critical): " << time_per_iteration_critical << " seconds\n";
    output_file << "Total execution time (atomic): " << total_time_atomic << " seconds\n";
    output_file << "Time per iteration (atomic): " << time_per_iteration_atomic << " seconds\n";
    output_file << "----------------------------------------\n";
}

int main() {
    // Open the file to save results
    std::ofstream output_file("parallel_main_openMP_results.txt");
    if (!output_file.is_open()) {
        std::cerr << "Error: Could not open the file to save results.\n";
        return 1;
    }

    // Define different configurations for normal runs
    std::vector<int> num_points_list = {100000, 250000, 500000, 1000000};
    std::vector<int> num_clusters_list = {5, 10, 20};
    int num_threads_default = 4; // Default number of threads

    // Run the algorithm for each configuration with both synchronization methods
    for (int num_points : num_points_list) {
        for (int num_clusters : num_clusters_list) {
            run_kmeans_with_both_methods(num_points, num_clusters, num_threads_default, output_file);
        }
    }

    // Special run for 1 million points, 20 clusters, with varying number of threads
    std::vector<int> thread_list = {2, 4, 6, 8, 10, 12, 14, 16};
    output_file << "Running special parallel K-means test for 1 million points with 20 clusters and varying thread counts.\n";
    for (int num_threads : thread_list) {
        run_kmeans_with_both_methods(1000000, 20, num_threads, output_file);
    }

    // Close the file
    output_file.close();

    return 0;
}
