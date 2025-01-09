// sequential_main.cpp
#include "common_functions.h"
#include "Cluster.h"
#include <iostream>
#include <fstream>
#include <chrono>
#include <vector>

// K-means algorithm with 20 fixed iterations
void kmeans(std::vector<Point>& points, std::vector<Cluster>& clusters, double& total_time) {
    const int max_iterations = 20;
    int iterations = 0;

    while (iterations < max_iterations) {
        auto start_iter = std::chrono::high_resolution_clock::now();

        // Step 1: Assign points to the nearest cluster
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


        // Step 2: Reset cluster values
        for (auto& cluster : clusters) {
            cluster.reset_values(); // Reset accumulated values
        }

        // Step 3: Accumulate points in clusters
        for (const auto& point : points) {
            clusters[point.get_cluster_id()].add_point(point);
        }

        // Step 4: Update centroids
        for (auto& cluster : clusters) {
            cluster.update_centroid(); // Update centroid coordinates
        }

        auto end_iter = std::chrono::high_resolution_clock::now();
        double iteration_time = std::chrono::duration<double>(end_iter - start_iter).count();  // Time in seconds
        total_time += iteration_time;  // Add to total time

        iterations++;
    }

    std::cout << "Completed " << max_iterations << " iterations.\n";
}

// Function to save the points and their clusters, and centroids to a CSV file
void save_to_csv(const std::vector<Point>& points, const std::vector<Cluster>& clusters, const std::string& filename) {
    std::ofstream csv_file(filename);

    // Write CSV headers
    csv_file << "x,y,cluster\n";

    // Write centroids first with a cluster ID of -1 (optional identifier)
    for (const auto& cluster : clusters) {
        csv_file << cluster.get_x_coord() << "," << cluster.get_y_coord() << ",-1\n";  // Use -1 for centroid identifier
    }

    // Write points and their cluster assignments
    for (const auto& point : points) {
        csv_file << point.get_x() << "," << point.get_y() << "," << point.get_cluster_id() << "\n";
    }

    csv_file.close();
}

// Function to run the K-means algorithm and save results to a file and CSV
void run_kmeans(int num_points, int num_clusters, std::ofstream& output_file) {
    std::cout << "Running K-means with " << num_points << " points and " << num_clusters << " clusters.\n";

    // Generate points and initialize clusters
    std::vector<Point> points = generate_points(num_points);
    std::vector<Cluster> clusters = initialize_clusters(num_clusters, points);

    // Measure total execution time
    double total_time = 0.0;

    kmeans(points, clusters, total_time);

    // Save points and clusters to CSV (including centroids)
    save_to_csv(points, clusters, "kmeans_results_" + std::to_string(num_clusters) + "_clusters.csv");

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
