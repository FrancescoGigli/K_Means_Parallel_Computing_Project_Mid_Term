#ifndef COMMON_FUNCTIONS_H
#define COMMON_FUNCTIONS_H

#include "Point.h"
#include "Cluster.h"
#include <vector>
#include <cstdlib>
#include <ctime>
#include <cmath>

// Add CUDA macro to support both GPU and CPU
#ifdef __CUDACC__
#define CUDA_HOST_DEVICE __host__ __device__
#else
#define CUDA_HOST_DEVICE
#endif

// Unified function to calculate the squared Euclidean distance
CUDA_HOST_DEVICE
inline float squared_euclidean_distance(const Point& pt, const Cluster& cluster) {
    float dx = pt.get_x() - cluster.get_x_coord();
    float dy = pt.get_y() - cluster.get_y_coord();
    return dx * dx + dy * dy;
}

// Generate random points within a fixed range
std::vector<Point> generate_points(int num_points, double range_min = 0.0, double range_max = 100.0) {
    std::vector<Point> points;
    points.reserve(num_points);

    // Fixed seed for reproducibility
    srand(42);

    for (int i = 0; i < num_points; ++i) {
        double x = range_min + static_cast<double>(rand()) / RAND_MAX * (range_max - range_min);
        double y = range_min + static_cast<double>(rand()) / RAND_MAX * (range_max - range_min);
        points.emplace_back(x, y);
    }
    return points;
}

// Initialize clusters by selecting the first points as centroids
std::vector<Cluster> initialize_clusters(int num_clusters, const std::vector<Point>& points) {
    std::vector<Cluster> clusters;
    clusters.reserve(num_clusters);
    for (int i = 0; i < num_clusters; ++i) {
        clusters.emplace_back(points[i].get_x(), points[i].get_y());
    }
    return clusters;
}

#endif // COMMON_FUNCTIONS_H
