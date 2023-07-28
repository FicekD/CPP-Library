#include <iostream>

#include "matrix.hpp"


int main() {
    ndarray::Matrix<int> x0(0, 5);
    std::cout << x0.size() << std::endl;
    ndarray::Matrix<int> x1(5, 5);
    x1.fill(3);
    ndarray::Matrix<int> x2(5, 5);
    x2.fill(6);
    ndarray::Matrix<int> x3 = x1 + x2;
    ndarray::Matrix<int> x4 = x3.pad(0, 2, 1, 5, 1);
    x4.at(2, 5) = 5;

    std::cout << x4.flip_lr() << std::endl;
    // std::cout << x4.reduce_sum() << std::endl;

    for (auto& x : x4) {
        std::cout << x << ' ';
    }
    std::cout << std::endl;

    std::cout << x4.reduce_sum() << std::endl;

    std::cout << ndarray::Matrix<int>::eye(9, 1) << std::endl;


    return 0;
}