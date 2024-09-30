#ifndef CLUSTER_H
#define CLUSTER_H

#ifdef __CUDACC__
#define CUDA_HOST_DEVICE __host__ __device__
#else
#define CUDA_HOST_DEVICE
#endif

#include "Point.h" // Ensure that Point is included

class Cluster {

private:
    double x_coord;        // X coordinate of the cluster's centroid
    double y_coord;        // Y coordinate of the cluster's centroid
    double new_x_coord;    // Accumulated X coordinate from points
    double new_y_coord;    // Accumulated Y coordinate from points
    int size;              // Number of points in the cluster

public:
    CUDA_HOST_DEVICE
    Cluster(double x_coord, double y_coord) {
        this->x_coord = x_coord;
        this->y_coord = y_coord;
        new_x_coord = 0;
        new_y_coord = 0;
        size = 0;
    }

    CUDA_HOST_DEVICE
    Cluster() {
        x_coord = 0;
        y_coord = 0;
        new_x_coord = 0;
        new_y_coord = 0;
        size = 0;
    }

    // Add a point to the cluster (accumulate coordinates)
    CUDA_HOST_DEVICE
    void add_point(const Point& pt) {
        new_x_coord += pt.get_x();  // Updated from get_x_coord() to get_x()
        new_y_coord += pt.get_y();  // Updated from get_y_coord() to get_y()
        size++;
    }

    // Reset the cluster for the next iteration
    CUDA_HOST_DEVICE
    void delete_values() {
        new_x_coord = 0;
        new_y_coord = 0;
        size = 0;
    }

    // Update the centroid's coordinates and return whether it changed
    CUDA_HOST_DEVICE
    bool update_values() {
        if (size > 0) {
            double new_x = new_x_coord / size;
            double new_y = new_y_coord / size;
            bool changed = (new_x != x_coord || new_y != y_coord);
            x_coord = new_x;
            y_coord = new_y;
            return changed;
        }
        return false;
    }

    // Get the current X coordinate of the centroid
    CUDA_HOST_DEVICE
    double get_x() const {
        return x_coord;
    }

    // Get the current Y coordinate of the centroid
    CUDA_HOST_DEVICE
    double get_y() const {
        return y_coord;
    }
};

#endif // CLUSTER_H
