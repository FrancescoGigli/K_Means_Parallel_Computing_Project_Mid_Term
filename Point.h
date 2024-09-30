#ifndef POINT_H
#define POINT_H

#ifdef __CUDACC__
#define CUDA_HOST_DEVICE __host__ __device__
#else
#define CUDA_HOST_DEVICE
#endif

class Point {

private:
    double coord_x; // X coordinate of the point
    double coord_y; // Y coordinate of the point
    int id_cluster; // ID of the cluster this point belongs to

public:
    CUDA_HOST_DEVICE
    Point(double coord_x, double coord_y) {
        this->coord_x = coord_x;
        this->coord_y = coord_y;
        id_cluster = 0;
    }

    CUDA_HOST_DEVICE
    Point() {
        coord_x = 0;
        coord_y = 0;
        id_cluster = 0;
    }

    CUDA_HOST_DEVICE
    double get_x() const {
        return coord_x;
    }

    CUDA_HOST_DEVICE
    double get_y() const {
        return coord_y;
    }

    CUDA_HOST_DEVICE
    void set_id(int id) {
        id_cluster = id;
    }

    CUDA_HOST_DEVICE
    int get_cluster_id() const {
        return id_cluster;
    }
};

#endif // POINT_H
