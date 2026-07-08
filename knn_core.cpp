#include<pybind11/pybind11.h>
#include<pybind11/numpy.h>
#include<cmath>

namespace py = pybind11;

double distance(py::array_t<double> a, py::array_t<double> b){
    auto buf_a = a.unchecked<1>();
    auto buf_b = b.unchecked<1>();

    double total = 0.0;
    for(int i = 0; i < buf_a.shape(0); i++){
        double diff = buf_a(i) - buf_b(i);
        total += diff * diff;
    }
    return std::sqrt(total);
}

PYBIND11_MODULE(knn_core, m){
    m.def("distance", &distance, "Euclidian distance between two images");
}