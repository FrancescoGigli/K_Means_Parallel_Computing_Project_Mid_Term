// Point.h
#ifndef POINT_H
#define POINT_H

#ifdef __CUDACC__
#define CUDA_HOST_DEVICE __host__ __device__
#else
#define CUDA_HOST_DEVICE
#endif

class Point {
private:
    double coord_x;    // X coordinate
    double coord_y;    // Y coordinate
    int id_cluster;    // Cluster ID

public:
    // Constructor with parameters
    CUDA_HOST_DEVICE
    Point(double coord_x, double coord_y) : coord_x(coord_x), coord_y(coord_y), id_cluster(0) {}

    // Default constructor
    CUDA_HOST_DEVICE
    Point() : coord_x(0.0), coord_y(0.0), id_cluster(0) {}

    // Getter for X coordinate
    CUDA_HOST_DEVICE
    double get_x() const { return coord_x; }

    // Getter for Y coordinate
    CUDA_HOST_DEVICE
    double get_y() const { return coord_y; }

    // Getter for Cluster ID
    CUDA_HOST_DEVICE
    int get_cluster_id() const { return id_cluster; }

    // Setter for Cluster ID
    CUDA_HOST_DEVICE
    void set_id(int id) { id_cluster = id; }
};

#endif // POINT_H
