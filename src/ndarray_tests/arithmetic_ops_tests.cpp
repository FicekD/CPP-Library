#include "CppUnitTest.h"

#include "../ndarray/matrix.hpp"

using namespace Microsoft::VisualStudio::CppUnitTestFramework;

namespace arithmetic_ops_tests {
	TEST_CLASS(arithmetic_ops_tests) {
	public:
		TEST_METHOD(UnaryPlus) {
			ndarray::Matrix<int> original_matrix(10, 10);
			original_matrix.fill(5);
			ndarray::Matrix<int> target_matrix = (+original_matrix);
			original_matrix.clear();
			Assert::IsTrue((target_matrix == 5).reduce_all());
		}
		TEST_METHOD(UnaryMinus) {
			ndarray::Matrix<int> original_matrix(10, 10);
			original_matrix.fill(5);
			ndarray::Matrix<int> target_matrix = (-original_matrix);
			original_matrix.clear();
			Assert::IsTrue((target_matrix == -5).reduce_all());
		}
		TEST_METHOD(Addition) {
			ndarray::Matrix<int> original_matrix_1(10, 10);
			original_matrix_1.fill(5);
			ndarray::Matrix<int> original_matrix_2(10, 10);
			original_matrix_2.fill(37);

			ndarray::Matrix<int> scalar_addition = original_matrix_1 + 8;
			ndarray::Matrix<int> elementwise_addition = original_matrix_1 + original_matrix_2;

			original_matrix_1.clear();
			original_matrix_2.clear();

			Assert::IsTrue((scalar_addition == 13).reduce_all());
			Assert::IsTrue((elementwise_addition == 42).reduce_all());
		}
		TEST_METHOD(Subtraction) {
			ndarray::Matrix<int> original_matrix_1(10, 10);
			original_matrix_1.fill(5);
			ndarray::Matrix<int> original_matrix_2(10, 10);
			original_matrix_2.fill(37);

			ndarray::Matrix<int> scalar_subtraction = original_matrix_1 - 8;
			ndarray::Matrix<int> elementwise_subtraction = original_matrix_1 - original_matrix_2;

			original_matrix_1.clear();
			original_matrix_2.clear();

			Assert::IsTrue((scalar_subtraction == -3).reduce_all());
			Assert::IsTrue((elementwise_subtraction == -32).reduce_all());
		}
		TEST_METHOD(Multiplication) {
			ndarray::Matrix<int> original_matrix_1(10, 10);
			original_matrix_1.fill(5);
			ndarray::Matrix<int> original_matrix_2(10, 10);
			original_matrix_2.fill(7);

			ndarray::Matrix<int> scalar_multiplication = original_matrix_1 * 8;
			ndarray::Matrix<int> elementwise_multiplication = original_matrix_1 * original_matrix_2;

			original_matrix_1.clear();
			original_matrix_2.clear();

			Assert::IsTrue((scalar_multiplication == 40).reduce_all());
			Assert::IsTrue((elementwise_multiplication == 35).reduce_all());
		}
		TEST_METHOD(Division) {
			ndarray::Matrix<int> original_matrix_1(10, 10);
			original_matrix_1.fill(40);
			ndarray::Matrix<int> original_matrix_2(10, 10);
			original_matrix_2.fill(20);

			ndarray::Matrix<int> scalar_division = original_matrix_1 / 8;
			ndarray::Matrix<int> elementwise_division = original_matrix_1 / original_matrix_2;

			original_matrix_1.clear();
			original_matrix_2.clear();

			Assert::IsTrue((scalar_division == 5).reduce_all());
			Assert::IsTrue((elementwise_division == 2).reduce_all());
		}
		TEST_METHOD(Square) {
			ndarray::Matrix<int> original_matrix(10, 10);
			original_matrix.fill(5);
			ndarray::Matrix<int> target_matrix = original_matrix.square();
			original_matrix.clear();
			Assert::IsTrue((target_matrix == 25).reduce_all());
		}
		TEST_METHOD(SquareRoot) {
			ndarray::Matrix<int> original_matrix(10, 10);
			original_matrix.fill(49);
			ndarray::Matrix<int> target_matrix = original_matrix.sqrt();
			original_matrix.clear();
			Assert::IsTrue((target_matrix == 7).reduce_all());
		}
		TEST_METHOD(Power) {
			ndarray::Matrix<int> original_matrix(10, 10);
			original_matrix.fill(3);
			ndarray::Matrix<int> target_matrix = original_matrix.pow(3);
			original_matrix.clear();
			Assert::IsTrue((target_matrix == 27).reduce_all());
		}
	};
	TEST_CLASS(arithmetic_inplace_ops_tests) {
	public:
		TEST_METHOD(Addition) {
			ndarray::Matrix<int> original_matrix(10, 10);
			original_matrix.fill(5);
			ndarray::Matrix<int> matrix(10, 10);
			matrix.fill(37);

			original_matrix.add_inplace(8);
			Assert::IsTrue((original_matrix == 13).reduce_all());

			original_matrix.add_inplace(17);
			matrix.clear();

			Assert::IsTrue((original_matrix == 30).reduce_all());
		}
		TEST_METHOD(Subtraction) {
			ndarray::Matrix<int> original_matrix(10, 10);
			original_matrix.fill(5);
			ndarray::Matrix<int> matrix(10, 10);
			matrix.fill(37);

			original_matrix.subtract_inplace(8);
			Assert::IsTrue((original_matrix == -3).reduce_all());

			original_matrix.subtract_inplace(17);
			matrix.clear();

			Assert::IsTrue((original_matrix == -20).reduce_all());
		}
		TEST_METHOD(Multiplication) {
			ndarray::Matrix<int> original_matrix(10, 10);
			original_matrix.fill(5);
			ndarray::Matrix<int> matrix(10, 10);
			matrix.fill(37);

			original_matrix.multiply_inplace(8);
			Assert::IsTrue((original_matrix == 40).reduce_all());

			original_matrix.multiply_inplace(2);
			matrix.clear();

			Assert::IsTrue((original_matrix == 80).reduce_all());
		}
		TEST_METHOD(Division) {
			ndarray::Matrix<int> original_matrix(10, 10);
			original_matrix.fill(40);
			ndarray::Matrix<int> matrix(10, 10);
			matrix.fill(37);

			original_matrix.divide_inplace(8);
			Assert::IsTrue((original_matrix == 5).reduce_all());

			original_matrix.divide_inplace(5);
			matrix.clear();

			Assert::IsTrue((original_matrix == 1).reduce_all());
		}
		TEST_METHOD(Square) {
			ndarray::Matrix<int> original_matrix(10, 10);
			original_matrix.fill(3);
			original_matrix.square_inplace();
			Assert::IsTrue((original_matrix == 9).reduce_all());
		}
		TEST_METHOD(SquareRoot) {
			ndarray::Matrix<int> original_matrix(10, 10);
			original_matrix.fill(81);
			original_matrix.sqrt_inplace();
			Assert::IsTrue((original_matrix == 9).reduce_all());
		}
		TEST_METHOD(Power) {
			ndarray::Matrix<int> original_matrix(10, 10);
			original_matrix.fill(3);
			original_matrix.pow_inplace(3);
			Assert::IsTrue((original_matrix == 27).reduce_all());
		}
	};
}