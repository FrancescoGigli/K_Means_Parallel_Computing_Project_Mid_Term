// Cluster.h
#ifndef CLUSTER_H
#define CLUSTER_H

#include "Point.h"

#ifdef __CUDACC__
#define CUDA_HOST_DEVICE __host__ __device__
#else
#define CUDA_HOST_DEVICE
#endif

class Cluster {
private:
    double x_coord;        // Centroid's X coordinate
    double y_coord;        // Centroid's Y coordinate
    double new_x_coord;    // Sum of X coordinates of assigned points
    double new_y_coord;    // Sum of Y coordinates of assigned points
    int size;              // Number of points in the cluster

public:
    // Constructor with parameters
    CUDA_HOST_DEVICE
    Cluster(double x_coord, double y_coord)
            : x_coord(x_coord), y_coord(y_coord), new_x_coord(0.0), new_y_coord(0.0), size(0) {}

    // Default constructor
    CUDA_HOST_DEVICE
    Cluster()
            : x_coord(0.0), y_coord(0.0), new_x_coord(0.0), new_y_coord(0.0), size(0) {}

    // Getter methods
    CUDA_HOST_DEVICE double get_x_coord() const { return x_coord; }
    CUDA_HOST_DEVICE double get_y_coord() const { return y_coord; }
    CUDA_HOST_DEVICE double get_new_x_coord() const { return new_x_coord; }
    CUDA_HOST_DEVICE double get_new_y_coord() const { return new_y_coord; }
    CUDA_HOST_DEVICE int get_size() const { return size; }

    // Setter methods
    CUDA_HOST_DEVICE void set_x_coord(double value) { x_coord = value; }
    CUDA_HOST_DEVICE void set_y_coord(double value) { y_coord = value; }

    // Reset accumulated values
    CUDA_HOST_DEVICE void reset_values() {
        new_x_coord = 0.0;
        new_y_coord = 0.0;
        size = 0;
    }

    // Methods for CUDA (if needed)
#ifdef __CUDACC__
    __device__ void atomic_add_to_new_x_coord(double value) {
        atomicAdd(&new_x_coord, value);
    }

    __device__ void atomic_add_to_new_y_coord(double value) {
        atomicAdd(&new_y_coord, value);
    }

    __device__ void atomic_increment_size() {
        atomicAdd(&size, 1);
    }
#endif

#ifndef __CUDACC__
    // Add a point to the cluster (non-atomic)
    void add_point(const Point& pt) {
        new_x_coord += pt.get_x();
        new_y_coord += pt.get_y();
        size++;
    }

    // Add a point to the cluster using atomic operations (OpenMP)
    void add_point_atomic(const Point& pt) {
#pragma omp atomic
        new_x_coord += pt.get_x();
#pragma omp atomic
        new_y_coord += pt.get_y();
#pragma omp atomic
        size++;
    }
#endif

    // Update the centroid's coordinates
    void update_centroid() {
        if (size > 0) {
            x_coord = new_x_coord / size;
            y_coord = new_y_coord / size;
        }
    }
};

#endif // CLUSTER_H
