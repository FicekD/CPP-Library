#include <iostream>

#include "matrix.hpp"
#include "linalg.hpp"
#include "ndarray.hpp"

using namespace ndarray;


int main() {
    Matrix<double> x0(0, 5);
    std::cout << x0.size() << std::endl;
    Matrix<double> x1(5, 5);
    x1.fill(3);
    Matrix<double> x2(5, 5);
    x2.fill(6);
    Matrix<double> x3 = x1 + x2;
    Matrix<double> x4 = x3.pad(0, 2, 1, 5, 1);
    x4.at(2, 5) = 5;

    transpose(x0);

    std::cout << x4.flip_lr() << std::endl;
    // std::cout << x4.reduce_sum() << std::endl;

    for (auto& x : x4) {
        std::cout << x << ' ';
    }
    std::cout << std::endl;

    std::cout << x4.reduce_sum() << std::endl;
    std::cout << x4.reduce_sum(COLS) << std::endl;

    std::cout << Matrix<double>::eye(9, 1) << std::endl;

    Matrix<double> x5(16, 1, { -0.0, -0.0, -596.59, -0.0, -0.0, -591.58, -780.01, -0.0, -248.83, -0.0, -0.0, -928.11, -68.80, -85.28, -426.91, -181.66 });

    auto fn = [](Matrix<double>& m1) { m1.square_inplace(); };
    std::cout << x4 << std::endl;
    std::cout << x4.square() << std::endl;
    fn(x4);
    std::cout << x4 << std::endl;

    return 0;
}