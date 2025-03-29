#include <iostream>

#include "../ndarray/matrix.hpp"
#include "../ndarray/linalg.hpp"
#include "../ndarray/ndarray.hpp"

using namespace ndarray;


int main() {
    /*
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
    std::cout << x4.reduce_sum(MatrixDim::COLS) << std::endl;

    std::cout << Matrix<double>::eye(9, 1) << std::endl;

    Matrix<double> x5(16, 1, { -0.0, -0.0, -596.59, -0.0, -0.0, -591.58, -780.01, -0.0, -248.83, -0.0, -0.0, -928.11, -68.80, -85.28, -426.91, -181.66 });

    auto fn = [](Matrix<double>& m1) { m1.square_inplace(); };
    std::cout << x4 << std::endl;
    std::cout << x4.square() << std::endl;
    fn(x4);
    std::cout << x4 << std::endl;

    Matrix<float> x6(5, 5);
    std::cout << x6 << std::endl;
    x6.fill(3);
    x6 = x6.reduce_sum(MatrixDim::ROWS);
    std::cout << x6 << std::endl;

    Matrix<bool> x7(5, 5);
    std::cout << x7 << std::endl;
    std::cout << x7.reduce_any(MatrixDim::COLS) << std::endl;
    

    std::size_t dim = 10;
    std::size_t stride = 2;

    ndarray::Matrix<std::size_t> matrix = ndarray::Matrix<std::size_t>(dim, dim);
    ndarray::Matrix<std::size_t> matrix_view = matrix.view(0, stride, matrix.rows(), 0, 1, matrix.cols());
    // std::cout << matrix_view << std::endl;
    for (std::size_t i = 0; i < 3; i += stride) {
        matrix_view = matrix_view.view(1, matrix_view.rows(), 1, matrix_view.cols());
        // matrix_view.at(0, 0) = 1;
        matrix_view.fill(i + 1);
    }
    std::cout << matrix << std::endl;
    */

    ndarray::Matrix<int> mat1(7, 5);
    ndarray::Matrix<int> mat2(7, 2);
    ndarray::Matrix<int> mat3(7, 4);

    mat1.fill(1);
    mat2.fill(2);
    mat3.fill(3);

    std::cout << mat1 << std::endl;
    std::cout << mat2 << std::endl;
    std::cout << mat3 << std::endl;

    std::cout << "Vector" << std::endl;
    std::vector<const ndarray::Matrix<int>*> mats { &mat1, &mat2, &mat3 };

    std::cout << "Concat" << std::endl;
    ndarray::Matrix<int> concatted(mats, ndarray::COLS);

    std::cout << concatted << std::endl;

    std::cout << "Submats" << std::endl;
    ndarray::Matrix<int> submat1 = concatted.view(0, mat1.rows(), 0, mat1.cols());
    std::cout << submat1 << std::endl;
    ndarray::Matrix<int> submat2 = concatted.view(0, mat1.rows(), mat1.cols(), mat1.cols() + mat2.cols());
    std::cout << submat2 << std::endl;
    ndarray::Matrix<int> submat3 = concatted.view(0, mat1.rows(), mat1.cols() + mat2.cols(), mat1.cols() + mat2.cols() + mat3.cols());
    std::cout << submat3 << std::endl;

    return 0;
}