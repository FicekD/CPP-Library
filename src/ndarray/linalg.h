#pragma once

#include "matrix.hpp"

template <typename T>
ndarray::Matrix<T> dot(const ndarray::Matrix<T>& matrix_1, const ndarray::Matrix<T>& matrix_2);

template <typename T>
ndarray::Matrix<T> transpose(const ndarray::Matrix<T>& matrix);

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
T trace(const ndarray::Matrix<T>& matrix, int k = 0);
