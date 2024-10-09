#ifndef COMMON_FUNCTIONS_H
#define COMMON_FUNCTIONS_H

#include <vector>
#include <random>
#include <cmath>       // For pow()
#include "Point.h"
#include "Cluster.h"   // Ensure Point and Cluster are available

// Function to generate random points
std::vector<Point> generate_points(int num_points) {
    std::vector<Point> points;

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0.0, 10000.0);

    for (int i = 0; i < num_points; ++i) {
        double x = dis(gen);
        double y = dis(gen);
        points.emplace_back(x, y);
    }
    return points;
}

// Function to initialize clusters with random centroids
std::vector<Cluster> initialize_clusters(int num_clusters, const std::vector<Point>& points) {
    std::vector<Cluster> clusters;

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, points.size() - 1);

    for (int i = 0; i < num_clusters; ++i) {
        int random_index = dis(gen);
        clusters.emplace_back(points[random_index].get_x(), points[random_index].get_y());
    }
    return clusters;
}

// Function to calculate squared Euclidean distance (no sqrt needed for comparisons)
CUDA_HOST_DEVICE
double squared_euclidean_distance(const Point& pt, const Cluster& cl) {
    double dx = pt.get_x() - cl.get_x_coord();
    double dy = pt.get_y() - cl.get_y_coord();
    return dx * dx + dy * dy;
}

#endif // COMMON_FUNCTIONS_H
