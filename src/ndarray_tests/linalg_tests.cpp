#include <catch2/catch_test_macros.hpp>

#include "../ndarray/matrix.hpp"
#include "../ndarray/linalg.hpp"

#include "generated_tests_io.hpp"
#include "general_tests.hpp"

using namespace generated_tests;

namespace linalg_tests {
	struct MatrixFixture {
		std::vector<ndarray::Matrix<double>> inputs;

		MatrixFixture() {
			read_inputs(tests_path + "inputs.bin", inputs);
		}
	};

	TEST_CASE_METHOD(MatrixFixture, "dot", "[ndarray][matrix][linalg]") {
		std::vector<TestOutput<double>> outputs;
		read_outputs(tests_path + "LinearAlgebra/dot.bin", outputs);
		run_generated_test<double>(inputs, outputs, [](const ndarray::Matrix<double>& m1, const ndarray::Matrix<double>& m2) { return ndarray::dot(m1, m2); }, true);
	}
	TEST_CASE_METHOD(MatrixFixture, "transpose", "[ndarray][matrix][linalg]") {
		std::vector<TestOutput<double>> outputs;
		read_outputs(tests_path + "LinearAlgebra/transpose.bin", outputs);
		run_generated_test<double>(inputs, outputs, [](const ndarray::Matrix<double>& m1) { return ndarray::transpose(m1); });
	}
	TEST_CASE_METHOD(MatrixFixture, "inverse", "[ndarray][matrix][linalg]") {
		std::vector<TestOutput<double>> outputs;
		read_outputs(tests_path + "LinearAlgebra/inverse.bin", outputs);
		run_generated_test<double>(inputs, outputs, [](const ndarray::Matrix<double>& m1) { return ndarray::inverse(m1); }, true);
	}
	TEST_CASE_METHOD(MatrixFixture, "pseudoinverse", "[ndarray][matrix][linalg]") {
		std::vector<TestOutput<double>> outputs;
		read_outputs(tests_path + "LinearAlgebra/pseudoinverse.bin", outputs);
		run_generated_test<double>(inputs, outputs, [](const ndarray::Matrix<double>& m1) { return ndarray::pseudo_inverse(m1); }, true);
	}
	TEST_CASE_METHOD(MatrixFixture, "eigvals", "[ndarray][matrix][linalg]") {
		std::vector<TestOutput<double>> outputs;
		read_outputs(tests_path + "LinearAlgebra/eigenvalues.bin", outputs);
		run_generated_test<double>(inputs, outputs, [](const ndarray::Matrix<double>& m1) { return std::get<0>(ndarray::eig(m1)); }, true);
	}
	TEST_CASE_METHOD(MatrixFixture, "eigvectors", "[ndarray][matrix][linalg]") {
		std::vector<TestOutput<double>> outputs;
		read_outputs(tests_path + "LinearAlgebra/eigenvectors.bin", outputs);
		run_generated_test<double>(inputs, outputs, [](const ndarray::Matrix<double>& m1) { return std::get<1>(ndarray::eig(m1)); }, true);
	}
	TEST_CASE_METHOD(MatrixFixture, "cholesky", "[ndarray][matrix][linalg]") {
		std::vector<TestOutput<double>> outputs;
		read_outputs(tests_path + "LinearAlgebra/cholesky.bin", outputs);
		run_generated_test<double>(inputs, outputs, [](const ndarray::Matrix<double>& m1) { return ndarray::cholesky(m1); }, true);
	}
	TEST_CASE_METHOD(MatrixFixture, "svd_u", "[ndarray][matrix][linalg]") {
		std::vector<TestOutput<double>> outputs;
		read_outputs(tests_path + "LinearAlgebra/svd-u.bin", outputs);
		run_generated_test<double>(inputs, outputs, [](const ndarray::Matrix<double>& m1) { return std::get<0>(ndarray::svd(m1)); }, true);
	}
	TEST_CASE_METHOD(MatrixFixture, "svd_sigma", "[ndarray][matrix][linalg]") {
		std::vector<TestOutput<double>> outputs;
		read_outputs(tests_path + "LinearAlgebra/svd-sigma.bin", outputs);
		run_generated_test<double>(inputs, outputs, [](const ndarray::Matrix<double>& m1) { return std::get<1>(ndarray::svd(m1)); }, true);
	}
	TEST_CASE_METHOD(MatrixFixture, "svd_v", "[ndarray][matrix][linalg]") {
		std::vector<TestOutput<double>> outputs;
		read_outputs(tests_path + "LinearAlgebra/svd-v.bin", outputs);
		run_generated_test<double>(inputs, outputs, [](const ndarray::Matrix<double>& m1) { return std::get<2>(ndarray::svd(m1)); }, true);
	}
	TEST_CASE_METHOD(MatrixFixture, "lu_l", "[ndarray][matrix][linalg]") {
		std::vector<TestOutput<double>> outputs;
		read_outputs(tests_path + "LinearAlgebra/lu-l.bin", outputs);
		run_generated_test<double>(inputs, outputs, [](const ndarray::Matrix<double>& m1) { return std::get<0>(ndarray::lu(m1)); }, true);
	}
	TEST_CASE_METHOD(MatrixFixture, "lu_u", "[ndarray][matrix][linalg]") {
		std::vector<TestOutput<double>> outputs;
		read_outputs(tests_path + "LinearAlgebra/lu-u.bin", outputs);
		run_generated_test<double>(inputs, outputs, [](const ndarray::Matrix<double>& m1) { return std::get<1>(ndarray::lu(m1)); }, true);
	}
	TEST_CASE_METHOD(MatrixFixture, "rank", "[ndarray][matrix][linalg]") {
		std::vector<TestOutput<double>> outputs;
		read_outputs(tests_path + "LinearAlgebra/rank.bin", outputs);
		run_generated_test<double>(inputs, outputs, [](const ndarray::Matrix<double>& m1) { return ndarray::Matrix<double>(1, 1, { (double)ndarray::rank(m1) }); });
	}
	TEST_CASE_METHOD(MatrixFixture, "determinant", "[ndarray][matrix][linalg]") {
		std::vector<TestOutput<double>> outputs;
		read_outputs(tests_path + "LinearAlgebra/determinant.bin", outputs);
		run_generated_test<double>(inputs, outputs, [](const ndarray::Matrix<double>& m1) { return ndarray::Matrix<double>(1, 1, { ndarray::det(m1) }); });
	}
	TEST_CASE_METHOD(MatrixFixture, "trace", "[ndarray][matrix][linalg]") {
		std::vector<TestOutput<double>> outputs;
		read_outputs(tests_path + "LinearAlgebra/trace.bin", outputs);
		run_generated_test<double>(inputs, outputs, [](const ndarray::Matrix<double>& m1) { return ndarray::Matrix<double>(1, 1, { ndarray::trace(m1) }); }, true);
	}
}