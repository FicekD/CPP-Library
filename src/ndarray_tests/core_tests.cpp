#include "CppUnitTest.h"

#include "../ndarray/matrix.hpp"

using namespace Microsoft::VisualStudio::CppUnitTestFramework;

namespace core_tests
{
	TEST_CLASS(core_tests) {
	public:
		TEST_METHOD(ConstructorDims) {
			ndarray::Matrix<int> matrix_square(10, 10);
			Assert::AreEqual(std::size_t(100), matrix_square.size());
			Assert::AreEqual(matrix_square.rows(), matrix_square.cols());

			ndarray::Matrix<int> matrix_row_vec(1, 10);
			ndarray::Matrix<int> matrix_col_vec(10, 1);
			Assert::AreEqual(std::size_t(1), matrix_row_vec.rows());
			Assert::AreEqual(std::size_t(10), matrix_row_vec.cols());
			Assert::AreEqual(matrix_row_vec.rows(), matrix_col_vec.cols());
			Assert::AreEqual(matrix_row_vec.cols(), matrix_col_vec.rows());

			ndarray::Matrix<int> matrix_empty(0, 10);
			Assert::AreEqual(std::size_t(0), matrix_empty.size());

			ndarray::Matrix<int> matrix_int(3, 7);
			ndarray::Matrix<bool> matrix_bool(3, 7);
			ndarray::Matrix<double> matrix_double(3, 7);
			Assert::AreEqual(matrix_int.size(), matrix_bool.size());
			Assert::AreEqual(matrix_int.size(), matrix_double.size());
		}
		TEST_METHOD(ConstructorFromArray) {
			int data[] = { 1,  2,  3,
						   4,  5,  6,
						   7,  8,  9,
						  10, 11, 12};
			
			ndarray::Matrix<int> matrix_row = ndarray::Matrix<int>(1, 12, data);
			for (int i = 0; i < matrix_row.size(); i++)
				Assert::AreEqual(data[i], matrix_row.get(i));
			
			ndarray::Matrix<int> matrix_4x3 = ndarray::Matrix<int>(4, 3, data);
			for (int row = 0; row < matrix_4x3.rows(); row++)
				for (int col = 0; col < matrix_4x3.cols(); col++) {
					std::size_t i = row * matrix_4x3.cols() + col;
					Assert::AreEqual(data[i], matrix_4x3.get(row, col));
				}
		}
		TEST_METHOD(ConstructorFromVector) {
			std::vector<int> data = { 1,  2,  3,
									  4,  5,  6,
									  7,  8,  9,
									 10, 11, 12 };

			ndarray::Matrix<int> matrix_row = ndarray::Matrix<int>(1, 12, data);
			for (int i = 0; i < matrix_row.size(); i++)
				Assert::AreEqual(data[i], matrix_row.get(i));

			ndarray::Matrix<int> matrix_4x3 = ndarray::Matrix<int>(4, 3, data);
			for (int row = 0; row < matrix_4x3.rows(); row++)
				for (int col = 0; col < matrix_4x3.cols(); col++) {
					std::size_t i = row * matrix_4x3.cols() + col;
					Assert::AreEqual(data[i], matrix_4x3.get(row, col));
				}
		}
		TEST_METHOD(ConstructorCopy) {
			ndarray::Matrix<int> matrix_origin = ndarray::Matrix<int>(4, 7);
			for (int i = 0; i < matrix_origin.size(); i++)
				matrix_origin.at(i) = i;

			ndarray::Matrix<int> matrix_copy = ndarray::Matrix<int>(matrix_origin);
			Assert::AreEqual(matrix_origin.rows(), matrix_copy.rows());
			Assert::AreEqual(matrix_origin.cols(), matrix_copy.cols());
			Assert::AreEqual(matrix_origin.size(), matrix_copy.size());
			for (int i = 0; i < matrix_origin.size(); i++)
				Assert::AreEqual(matrix_origin.get(i), matrix_copy.get(i));

			for (int i = 0; i < matrix_origin.size(); i++) {
				matrix_origin.at(i) += 1;
				Assert::AreNotEqual(matrix_origin.get(i), matrix_copy.get(i));
			}
		}
		TEST_METHOD(ConstructorMove) {
			const std::size_t rows = 4, cols = 7;
			ndarray::Matrix<int> matrix_origin = ndarray::Matrix<int>(rows, cols);
			for (int i = 0; i < matrix_origin.size(); i++)
				matrix_origin.at(i) = i;
			
			ndarray::Matrix<int> matrix_move = std::move(matrix_origin);

			Assert::AreEqual(std::size_t(0), matrix_origin.rows());
			Assert::AreEqual(std::size_t(0), matrix_origin.cols());
			Assert::AreEqual(std::size_t(0), matrix_origin.size());
			Assert::AreEqual(rows, matrix_move.rows());
			Assert::AreEqual(cols, matrix_move.cols());
			Assert::AreEqual(rows * cols, matrix_move.size());
			for (int i = 0; i < matrix_origin.size(); i++)
				Assert::AreEqual(i, matrix_move.get(i));
		}
		TEST_METHOD(ConstructorRowConcat) {
			ndarray::Matrix<int> mat1(5, 7);
			ndarray::Matrix<int> mat2(2, 7);
			ndarray::Matrix<int> mat3(4, 7);

			ndarray::Matrix<int> concatted(std::vector<ndarray::Matrix<int>> { mat1, mat2, mat3 }, ndarray::ROWS);

			Assert::AreEqual(mat1.rows() + mat2.rows() + mat3.rows(), concatted.rows());
			Assert::AreEqual(mat1.cols(), concatted.cols());

			ndarray::Matrix<int> submat1 = concatted.sub_matrix(0, mat1.rows(), 0, mat1.cols());
			ndarray::Matrix<int> submat2 = concatted.sub_matrix(mat1.rows(), mat1.rows() + mat2.rows(), 0, mat1.cols());
			ndarray::Matrix<int> submat3 = concatted.sub_matrix(mat1.rows() + mat2.rows(), mat1.rows() + mat2.rows() + mat3.rows(), 0, mat1.cols());

			Assert::IsTrue((mat1 == submat1).reduce_all());
			Assert::IsTrue((mat2 == submat2).reduce_all());
			Assert::IsTrue((mat3 == submat3).reduce_all());
		}
		TEST_METHOD(ConstructorColConcat) {
			ndarray::Matrix<int> mat1(7, 5);
			ndarray::Matrix<int> mat2(7, 2);
			ndarray::Matrix<int> mat3(7, 4);

			ndarray::Matrix<int> concatted(std::vector<ndarray::Matrix<int>> { mat1, mat2, mat3 }, ndarray::COLS);

			Assert::AreEqual(mat1.cols() + mat2.cols() + mat3.cols(), concatted.cols());
			Assert::AreEqual(mat1.rows(), concatted.rows());

			ndarray::Matrix<int> submat1 = concatted.sub_matrix(0, mat1.rows(), 0, mat1.cols());
			ndarray::Matrix<int> submat2 = concatted.sub_matrix(0, mat1.rows(), mat1.cols(), mat1.cols() + mat2.cols());
			ndarray::Matrix<int> submat3 = concatted.sub_matrix(0, mat1.rows(), mat1.cols() + mat2.cols(), mat1.cols() + mat2.cols() + mat3.cols());

			Assert::IsTrue((mat1 == submat1).reduce_all());
			Assert::IsTrue((mat2 == submat2).reduce_all());
			Assert::IsTrue((mat3 == submat3).reduce_all());
		}
		TEST_METHOD(ConstructorStaticEye) {
			std::size_t size = 9;
			ndarray::Matrix<int> eye_k_0 = ndarray::Matrix<int>::eye(size, 0);
			ndarray::Matrix<int> eye_k_1 = ndarray::Matrix<int>::eye(size, 1);
			ndarray::Matrix<int> eye_k_neg_1 = ndarray::Matrix<int>::eye(size, -1);
			ndarray::Matrix<int> eye_k_size = ndarray::Matrix<int>::eye(size, int(size) - 1);
			
			Assert::AreEqual(std::size_t(size), eye_k_0.rows());
			Assert::AreEqual(std::size_t(size), eye_k_0.cols());
			Assert::AreEqual(std::size_t(size * size), eye_k_0.size());
			for (int row = 0; row < eye_k_0.rows(); row++)
				for (int col = 0; col < eye_k_0.cols(); col++) {
					Assert::AreEqual(int(row == col), eye_k_0.get(row, col));
					Assert::AreEqual(int(row == col - 1), eye_k_1.get(row, col));
					Assert::AreEqual(int(row == col - (size - 1)), eye_k_size.get(row, col));
					Assert::AreEqual(int(row == col + 1), eye_k_neg_1.get(row, col));
				}
		}
		TEST_METHOD(AtGet) {
			ndarray::Matrix<int> matrix(5, 3);
			for (int i = 0; i < matrix.size(); i++)
				Assert::AreEqual(0, matrix.get(i));
			
			for (int i = 0; i < matrix.size(); i++) {
				matrix.at(i) = -1;
				Assert::AreEqual(-1, matrix.get(i));
			}
			
			for (int row = 0; row < matrix.rows(); row++)
				for (int col = 0; col < matrix.cols(); col++)
					matrix.at(row, col) = -2;
			for (int i = 0; i < matrix.size(); i++)
				Assert::AreEqual(-2, matrix.get(i));
			
			matrix.at(4, 2) = 42;
			Assert::AreEqual(42, matrix.get(4, 2));
		}
		TEST_METHOD(Clear) {
			ndarray::Matrix<int> matrix = ndarray::Matrix<int>(42, 42);
			matrix.clear();

			Assert::AreEqual(std::size_t(0), matrix.rows());
			Assert::AreEqual(std::size_t(0), matrix.cols());
			Assert::AreEqual(std::size_t(0), matrix.size());
		}
		TEST_METHOD(Fill) {
			ndarray::Matrix<int> matrix = ndarray::Matrix<int>(42, 42);
			matrix.fill(42);
			for (int i = 0; i < matrix.size(); i++)
				Assert::AreEqual(42, matrix.get(i));
		}
	};
}
