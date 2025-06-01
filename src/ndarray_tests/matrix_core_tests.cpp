#include <catch2/catch_test_macros.hpp>

#include "../ndarray/matrix.hpp"

namespace matrix_core_tests
{
	TEST_CASE("constructordims", "[ndarray][matrix][core]") {
		ndarray::Matrix<int> matrix_square(10, 10);
		REQUIRE(std::size_t(100) == matrix_square.size());
		REQUIRE(matrix_square.rows() == matrix_square.cols());

		ndarray::Matrix<int> matrix_row_vec(1, 10);
		ndarray::Matrix<int> matrix_col_vec(10, 1);
		REQUIRE(std::size_t(1) == matrix_row_vec.rows());
		REQUIRE(std::size_t(10) == matrix_row_vec.cols());
		REQUIRE(matrix_row_vec.rows() == matrix_col_vec.cols());
		REQUIRE(matrix_row_vec.cols() == matrix_col_vec.rows());

		ndarray::Matrix<int> matrix_empty(0, 10);
		REQUIRE(std::size_t(0) == matrix_empty.size());

		ndarray::Matrix<int> matrix_int(3, 7);
		ndarray::Matrix<bool> matrix_bool(3, 7);
		ndarray::Matrix<double> matrix_double(3, 7);
		REQUIRE(matrix_int.size() == matrix_bool.size());
		REQUIRE(matrix_int.size() == matrix_double.size());
	}
	TEST_CASE("constructorfromarray", "[ndarray][matrix][core]") {
		int data[] = { 1,  2,  3,
					   4,  5,  6,
					   7,  8,  9,
					   10, 11, 12};
			
		ndarray::Matrix<int> matrix_row = ndarray::Matrix<int>(1, 12, data);
		for (int i = 0; i < matrix_row.size(); i++)
			REQUIRE(data[i] == matrix_row.get(i));
			
		ndarray::Matrix<int> matrix_4x3 = ndarray::Matrix<int>(4, 3, data);
		for (int row = 0; row < matrix_4x3.rows(); row++)
			for (int col = 0; col < matrix_4x3.cols(); col++) {
				std::size_t i = row * matrix_4x3.cols() + col;
				REQUIRE(data[i] == matrix_4x3.get(row, col));
			}
	}
	TEST_CASE("constructorfromvector", "[ndarray][matrix][core]") {
		std::vector<int> data = { 1,  2,  3,
								  4,  5,  6,
								  7,  8,  9,
								  10, 11, 12 };

		ndarray::Matrix<int> matrix_row = ndarray::Matrix<int>(1, 12, data);
		for (int i = 0; i < matrix_row.size(); i++)
			REQUIRE(data[i] == matrix_row.get(i));

		ndarray::Matrix<int> matrix_4x3 = ndarray::Matrix<int>(4, 3, data);
		for (int row = 0; row < matrix_4x3.rows(); row++)
			for (int col = 0; col < matrix_4x3.cols(); col++) {
				std::size_t i = row * matrix_4x3.cols() + col;
				REQUIRE(data[i] == matrix_4x3.get(row, col));
			}
	}
	TEST_CASE("constructorcopy", "[ndarray][matrix][core]") {
		ndarray::Matrix<int> matrix_origin = ndarray::Matrix<int>(4, 7);
		for (int i = 0; i < matrix_origin.size(); i++)
			matrix_origin.at(i) = i;

		ndarray::Matrix<int> matrix_copy = ndarray::Matrix<int>(matrix_origin);
		REQUIRE(matrix_origin.rows() == matrix_copy.rows());
		REQUIRE(matrix_origin.cols() == matrix_copy.cols());
		REQUIRE(matrix_origin.size() == matrix_copy.size());
		for (int i = 0; i < matrix_origin.size(); i++)
			REQUIRE(matrix_origin.get(i) == matrix_copy.get(i));

		for (int i = 0; i < matrix_origin.size(); i++) {
			matrix_origin.at(i) += 1;
			REQUIRE(matrix_origin.get(i) != matrix_copy.get(i));
		}
	}
	TEST_CASE("constructormove", "[ndarray][matrix][core]") {
		const std::size_t rows = 4, cols = 7;
		ndarray::Matrix<int> matrix_origin = ndarray::Matrix<int>(rows, cols);
		for (int i = 0; i < matrix_origin.size(); i++)
			matrix_origin.at(i) = i;
			
		ndarray::Matrix<int> matrix_move = std::move(matrix_origin);

#pragma warning(push)
#pragma warning(disable: 26800)
		REQUIRE(std::size_t(0) == matrix_origin.rows());
		REQUIRE(std::size_t(0) == matrix_origin.cols());
		REQUIRE(std::size_t(0) == matrix_origin.size());
#pragma warning(pop)

		REQUIRE(rows == matrix_move.rows());
		REQUIRE(cols == matrix_move.cols());
		REQUIRE(rows * cols == matrix_move.size());
		for (int i = 0; i < matrix_origin.size(); i++)
			REQUIRE(i == matrix_move.get(i));
	}
	TEST_CASE("constructorrowconcat", "[ndarray][matrix][core]") {
		ndarray::Matrix<int> mat1(5, 7);
		ndarray::Matrix<int> mat2(2, 7);
		ndarray::Matrix<int> mat3(4, 7);

		mat1.fill(1);
		mat2.fill(2);
		mat3.fill(3);

		ndarray::Matrix<int> concatted(std::vector<const ndarray::Matrix<int>*> { &mat1, &mat2, &mat3 }, ndarray::MatrixDim::ROWS);

		REQUIRE(mat1.rows() + mat2.rows() + mat3.rows() == concatted.rows());
		REQUIRE(mat1.cols() == concatted.cols());

		ndarray::Matrix<int> submat1 = concatted.view(0, mat1.rows(), 0, mat1.cols());
		ndarray::Matrix<int> submat2 = concatted.view(mat1.rows(), mat1.rows() + mat2.rows(), 0, mat1.cols());
		ndarray::Matrix<int> submat3 = concatted.view(mat1.rows() + mat2.rows(), mat1.rows() + mat2.rows() + mat3.rows(), 0, mat1.cols());

		REQUIRE((mat1 == submat1).reduce_all());
		REQUIRE((mat2 == submat2).reduce_all());
		REQUIRE((mat3 == submat3).reduce_all());
	}
	TEST_CASE("constructorcolconcat", "[ndarray][matrix][core]") {
		ndarray::Matrix<int> mat1(7, 5);
		ndarray::Matrix<int> mat2(7, 2);
		ndarray::Matrix<int> mat3(7, 4);

		mat1.fill(1);
		mat2.fill(2);
		mat3.fill(3);

		ndarray::Matrix<int> concatted(std::vector<const ndarray::Matrix<int>*> { &mat1, &mat2, &mat3 }, ndarray::COLS);

		REQUIRE(mat1.cols() + mat2.cols() + mat3.cols() == concatted.cols());
		REQUIRE(mat1.rows() == concatted.rows());

		ndarray::Matrix<int> submat1 = concatted.view(0, mat1.rows(), 0, mat1.cols());
		ndarray::Matrix<int> submat2 = concatted.view(0, mat1.rows(), mat1.cols(), mat1.cols() + mat2.cols());
		ndarray::Matrix<int> submat3 = concatted.view(0, mat1.rows(), mat1.cols() + mat2.cols(), mat1.cols() + mat2.cols() + mat3.cols());

		REQUIRE((mat1 == submat1).reduce_all());
		REQUIRE((mat2 == submat2).reduce_all());
		REQUIRE((mat3 == submat3).reduce_all());
	}
	TEST_CASE("constructorstaticeye", "[ndarray][matrix][core]") {
		std::size_t size = 9;
		ndarray::Matrix<int> eye_k_0 = ndarray::Matrix<int>::eye(size, 0);
		ndarray::Matrix<int> eye_k_1 = ndarray::Matrix<int>::eye(size, 1);
		ndarray::Matrix<int> eye_k_neg_1 = ndarray::Matrix<int>::eye(size, -1);
		ndarray::Matrix<int> eye_k_size = ndarray::Matrix<int>::eye(size, int(size) - 1);
			
		REQUIRE(std::size_t(size) == eye_k_0.rows());
		REQUIRE(std::size_t(size) == eye_k_0.cols());
		REQUIRE(std::size_t(size * size) == eye_k_0.size());
		for (int row = 0; row < eye_k_0.rows(); row++)
			for (int col = 0; col < eye_k_0.cols(); col++) {
				REQUIRE(int(row == col) == eye_k_0.get(row, col));
				REQUIRE(int(row == col - 1) == eye_k_1.get(row, col));
				REQUIRE(int(row == col - (size - 1)) == eye_k_size.get(row, col));
				REQUIRE(int(row == col + 1) == eye_k_neg_1.get(row, col));
			}
	}
	TEST_CASE("atget", "[ndarray][matrix][core]") {
		ndarray::Matrix<int> matrix(5, 3);
		for (int i = 0; i < matrix.size(); i++)
			REQUIRE(0 == matrix.get(i));
			
		for (int i = 0; i < matrix.size(); i++) {
			matrix.at(i) = -1;
			REQUIRE(-1 == matrix.get(i));
		}
			
		for (int row = 0; row < matrix.rows(); row++)
			for (int col = 0; col < matrix.cols(); col++)
				matrix.at(row, col) = -2;
		for (int i = 0; i < matrix.size(); i++)
			REQUIRE(-2 == matrix.get(i));
			
		matrix.at(4, 2) = 42;
		REQUIRE(42 == matrix.get(4, 2));
	}
	TEST_CASE("clear", "[ndarray][matrix][core]") {
		ndarray::Matrix<int> matrix = ndarray::Matrix<int>(42, 42);
		matrix.clear();

		REQUIRE(std::size_t(0) == matrix.rows());
		REQUIRE(std::size_t(0) == matrix.cols());
		REQUIRE(std::size_t(0) == matrix.size());
	}
	TEST_CASE("fill", "[ndarray][matrix][core]") {
		ndarray::Matrix<int> matrix = ndarray::Matrix<int>(42, 42);
		matrix.fill(42);
		for (int i = 0; i < matrix.size(); i++)
			REQUIRE(42 == matrix.get(i));
	}
	TEST_CASE("view", "[ndarray][matrix][core]") {
		std::size_t dim = 10;
		ndarray::Matrix<std::size_t> matrix = ndarray::Matrix<std::size_t>(dim, dim);
		ndarray::Matrix<std::size_t> matrix_view = matrix.view(0, matrix.rows(), 0, matrix.cols());
		for (std::size_t i = 0; i < dim - 1; i++) {
			matrix_view = matrix_view.view(1, matrix_view.rows(), 1, matrix_view.cols());
			matrix_view.fill(i + 1);
		}
		for (std::size_t row = 0; row < matrix.rows(); row++)
			for (std::size_t col = 0; col < matrix.cols(); col++)
				REQUIRE((row < col ? row : col) == matrix.get(row, col));

		matrix.fill(0);
		matrix_view = matrix.view(0, 2, matrix.rows(), 0, 1, matrix.cols());
		matrix_view.fill(1);
		matrix_view = matrix.view(1, 2, matrix.rows(), 0, 1, matrix.cols());
		matrix_view.fill(2);

		for (std::size_t row = 0; row < matrix.rows(); row++)
			for (std::size_t col = 0; col < matrix.cols(); col++)
				REQUIRE(std::size_t(row % 2 ? 2 : 1) == matrix.get(row, col));

		matrix.fill(0);
		matrix_view = matrix.view(0, 1, matrix.rows(), 0, 2, matrix.cols());
		matrix_view.fill(1);
		matrix_view = matrix.view(0, 1, matrix.rows(), 1, 2, matrix.cols());
		matrix_view.fill(2);

		for (std::size_t row = 0; row < matrix.rows(); row++)
			for (std::size_t col = 0; col < matrix.cols(); col++)
				REQUIRE(std::size_t(col % 2 ? 2 : 1) == matrix.get(row, col));

		matrix.fill(0);
		matrix_view = matrix.view(0, 2, matrix.rows(), 0, 2, matrix.cols());
		matrix_view.fill(1);

		for (std::size_t row = 0; row < matrix.rows(); row++)
			for (std::size_t col = 0; col < matrix.cols(); col++)
				REQUIRE(std::size_t(row % 2 || col % 2 ? 0 : 1) == matrix.get(row, col));
	}
}
