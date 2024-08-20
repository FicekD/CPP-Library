#include "linalg.h"

template <typename T>
ndarray::Matrix<T> dot(const ndarray::Matrix<T>& matrix_1, const ndarray::Matrix<T>& matrix_2) {
    if (matrix_1.cols() != matrix_2.rows())
        throw std::invalid_argument("Matricies shape missmatch");

    ndarray::Matrix<T> result(matrix_1.rows(), matrix_2.cols());
    for (std::size_t row = 0; row < matrix_1.rows(); row++) {
        for (std::size_t col = 0; col < matrix_2.cols(); col++) {
            T sum = T(0);
            for (std::size_t i = 0; i < matrix_1.cols(); i++) {
                sum += this->at(row, i) * matrix_2.at(i, col);
            }
            result.at(row, col) = sum;
        }
    }
    return result;
}

template <typename T>
ndarray::Matrix<T> transpose(const ndarray::Matrix<T>& matrix) {
    ndarray::Matrix<T> mat(matrix.cols(), matrix.rows());
    for (std::size_t row = 0; row < matrix.rows(); row++) {
        for (std::size_t col = 0; col < matrix.cols(); col++) {
            mat.at(col, row) = this->at(row, col);
        }
    }
    return mat;
}

template <typename T>
ndarray::Matrix<T> inverse(const ndarray::Matrix<T>& matrix);

template <typename T>
ndarray::Matrix<T> pseudo_inverse();

template <typename T>
std::tuple<ndarray::Matrix<T>, ndarray::Matrix<T>> eig();

template <typename T>
ndarray::Matrix<T> cholesky(const ndarray::Matrix<T>& matrix);

template <typename T>
std::tuple<ndarray::Matrix<T>, ndarray::Matrix<T>, ndarray::Matrix<T>> svd(const ndarray::Matrix<T>& matrix);

template <typename T>
std::size_t rank(const ndarray::Matrix<T>& matrix);

template <typename T>
T det(const ndarray::Matrix<T>& matrix);

template <typename T>
T trace(const ndarray::Matrix<T>& matrix, int k);