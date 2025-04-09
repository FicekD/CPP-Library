#ifndef _LINALG_H
#define _LINALG_H

#include <numeric>

#include "matrix.hpp"

namespace ndarray {

	template <typename T>
	Matrix<T> dot(const Matrix<T>& matrix_1, const Matrix<T>& matrix_2) {
		if (matrix_1.cols() != matrix_2.rows())
			throw std::invalid_argument("Matricies shape missmatch");

		Matrix<T> result(matrix_1.rows(), matrix_2.cols());
		for (std::size_t row = 0; row < matrix_1.rows(); row++) {
			for (std::size_t col = 0; col < matrix_2.cols(); col++) {
				T sum = T(0);
				for (std::size_t i = 0; i < matrix_1.cols(); i++) {
					sum = std::fma(matrix_1.get(row, i), matrix_2.get(i, col), sum);
				}
				result.at(row, col) = sum;
			}
		}
		return result;
	}

	template <typename T>
	Matrix<T> transpose(const Matrix<T>& matrix) {
		Matrix<T> mat(matrix.cols(), matrix.rows());
		for (std::size_t row = 0; row < matrix.rows(); row++) {
			for (std::size_t col = 0; col < matrix.cols(); col++) {
				mat.at(col, row) = matrix.get(row, col);
			}
		}
		return mat;
	}

	namespace _inv {
		template <typename T>
		void multiply_row(Matrix<T>& matrix, std::size_t row, T multiplier) {
			for (std::size_t col = 0; col < matrix.cols(); col++) {
				T& ref = matrix.at(row, col);
				ref = ref * multiplier;
			}
		}

		template <typename T>
		std::size_t find_largest_abs_row_at_col(const Matrix<T>& matrix, std::size_t col) {
			std::pair<std::size_t, T> largest(col, T(0));
			for (std::size_t row = 0; row < matrix.rows(); row++) {
				T value = std::abs(matrix.get(row, col));
				if (value > largest.second) {
					largest.first = row;
					largest.second = value;
				}
			}
			return largest.first;
		}

		template <typename T>
		void subtract_multiplied_row(Matrix<T>& matrix, std::size_t target_row, std::size_t source_row, T multiplier) {
			for (std::size_t col = 0; col < matrix.cols(); col++) {
				T& ref = matrix.at(target_row, col);
				ref = std::fma(-multiplier, matrix.get(source_row, col), ref);
			}
		}
	}

	template <typename T>
	Matrix<T> inverse(const Matrix<T>& matrix) {
		if (matrix.cols() != matrix.rows())
			throw std::invalid_argument("Matrix has to be square");
		
		Matrix<T> eye = Matrix<T>::eye(matrix.rows());
		std::vector<const Matrix<T>*> vec { &matrix, &eye };
		Matrix<T> m(vec, MatrixDim::COLS);

		for (std::size_t col = 0; col < matrix.cols(); col++) {
			if (m.at(col, col) == 0) {
				std::size_t largest_row_idx = _inv::find_largest_abs_row_at_col(m, col);
				if (largest_row_idx == col) {
					throw std::invalid_argument("Matrix is singular");
				}
				m.swap_rows(col, largest_row_idx);
				// TODO: did I break previous pivot?
			}
		}

		for (std::size_t col = 0; col < matrix.cols() - 1; col++) {
			for (std::size_t row = col + 1; row < matrix.rows(); row++) {
				T k = m.get(row, col) / matrix.get(col, col);
				_inv::subtract_multiplied_row(m, row, col, k);
				m.at(row, col) = T(0);
			}
		}

		for (std::size_t row = 0; row < m.rows(); row++) {
			T multiplier = 1.0 / m.get(row, row);
			_inv::multiply_row(m, row, multiplier);
			m.at(row, row) = T(0);
		}

		for (std::size_t row = 0; row < m.rows(); row++) {
			for (std::size_t col = row + 1; col < matrix.cols(); col++) {
				T multiplier = m.get(row, col);
				_inv::subtract_multiplied_row(m, row, col, multiplier);
				m.at(row, col) = T(0);
			}
		}
		Matrix<T> inverse = m.view(0, m.rows(), matrix.cols(), m.cols());
		return inverse;
	}

	template <typename T>
	Matrix<T> pseudo_inverse(const Matrix<T>& matrix) {
		throw std::logic_error("Not implemented");
	}

	template <typename T>
	std::tuple<Matrix<T>, Matrix<T>> eig(const Matrix<T>& matrix) {
		throw std::logic_error("Not implemented");
	}

	template <typename T>
	Matrix<T> cholesky(const Matrix<T>& matrix) {
		if (matrix.cols() != matrix.rows())
			throw std::invalid_argument("Matrix has to be square");

		Matrix<T> result(matrix.rows(), matrix.rows());
		for (std::size_t j = 0; j < matrix.rows(); j++) {
			T sum = T(0);
			for (std::size_t k = 0; k < j; k++) {
				sum = std::fma(result.get(j, k), result.get(j, k), sum);
			}
			T diagonal_value = std::sqrt(matrix.get(j, j) - sum);
			result.at(j, j) = diagonal_value;

			for (std::size_t i = j + 1; i < matrix.rows(); i++) {
				sum = T(0);
				for (std::size_t k = 0; k < j; k++) {
					sum = std::fma(result.get(i, k), result.get(j, k), sum);
				}
				result.at(i, j) = ((matrix.get(i, j) - sum) / diagonal_value);
			}
		}
		return result;
	}

	template <typename T>
	std::tuple<Matrix<T>, Matrix<T>, Matrix<T>> svd(const Matrix<T>& matrix) {
		throw std::logic_error("Not implemented");
	}

	template <typename T>
	std::size_t rank(const Matrix<T>& matrix) {
		throw std::logic_error("Not implemented");
	}

	template <typename T>
	T det(const Matrix<T>& matrix) {
		throw std::logic_error("Not implemented");
	}

	template <typename T>
	T trace(const Matrix<T>& matrix, int k = 0) {
		throw std::logic_error("Not implemented");
	}
}

#endif