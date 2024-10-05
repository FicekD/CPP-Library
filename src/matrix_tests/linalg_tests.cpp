#include "CppUnitTest.h"

#include "../matrix/matrix.hpp"
#include "../matrix/linalg.hpp"

#include "generated_tests_io.hpp"
#include "general_tests.hpp"

using namespace Microsoft::VisualStudio::CppUnitTestFramework;
using namespace generated_tests;

namespace linalg_tests {
	std::vector<matrix::Matrix<double>> inputs;

	TEST_CLASS(LinearAlgebra) {
public:
	TEST_CLASS_INITIALIZE(ReadGeneratedTestDefinitions)
	{
		if (inputs.empty())
			read_inputs(tests_path + "inputs.bin", inputs);
	}
	TEST_METHOD(Dot) {
		std::vector<TestOutput<double>> outputs;
		read_outputs(tests_path + "LinearAlgebra/dot.bin", outputs);
		run_generated_test<double>(inputs, outputs, [](const matrix::Matrix<double>& m1, const matrix::Matrix<double>& m2) { return matrix::dot(m1, m2); });
	}
	TEST_METHOD(Transpose) {
		std::vector<TestOutput<double>> outputs;
		read_outputs(tests_path + "LinearAlgebra/transpose.bin", outputs);
		run_generated_test<double>(inputs, outputs, [](const matrix::Matrix<double>& m1) { return matrix::transpose(m1); });
	}
	TEST_METHOD(Inverse) {
		std::vector<TestOutput<double>> outputs;
		read_outputs(tests_path + "LinearAlgebra/inverse.bin", outputs);
		run_generated_test<double>(inputs, outputs, [](const matrix::Matrix<double>& m1) { return matrix::inverse(m1); });
	}
	TEST_METHOD(PseudoInverse) {
		std::vector<TestOutput<double>> outputs;
		read_outputs(tests_path + "LinearAlgebra/pseudoinverse.bin", outputs);
		run_generated_test<double>(inputs, outputs, [](const matrix::Matrix<double>& m1) { return matrix::pseudo_inverse(m1); });
	}
	TEST_METHOD(EigVals) {
		std::vector<TestOutput<double>> outputs;
		read_outputs(tests_path + "LinearAlgebra/eigenvalues.bin", outputs);
		run_generated_test<double>(inputs, outputs, [](const matrix::Matrix<double>& m1) { return std::get<0>(matrix::eig(m1)); });
	}
	TEST_METHOD(EigVectors) {
		std::vector<TestOutput<double>> outputs;
		read_outputs(tests_path + "LinearAlgebra/eigenvectors.bin", outputs);
		run_generated_test<double>(inputs, outputs, [](const matrix::Matrix<double>& m1) { return std::get<1>(matrix::eig(m1)); });
	}
	TEST_METHOD(Cholesky) {
		std::vector<TestOutput<double>> outputs;
		read_outputs(tests_path + "LinearAlgebra/cholesky.bin", outputs);
		run_generated_test<double>(inputs, outputs, [](const matrix::Matrix<double>& m1) { return matrix::cholesky(m1); });
	}
	TEST_METHOD(SVD_U) {
		std::vector<TestOutput<double>> outputs;
		read_outputs(tests_path + "LinearAlgebra/svd-u.bin", outputs);
		run_generated_test<double>(inputs, outputs, [](const matrix::Matrix<double>& m1) { return std::get<0>(matrix::svd(m1)); });
	}
	TEST_METHOD(SVD_Sigma) {
		std::vector<TestOutput<double>> outputs;
		read_outputs(tests_path + "LinearAlgebra/svd-sigma.bin", outputs);
		run_generated_test<double>(inputs, outputs, [](const matrix::Matrix<double>& m1) { return std::get<1>(matrix::svd(m1)); });
	}
	TEST_METHOD(SVD_V) {
		std::vector<TestOutput<double>> outputs;
		read_outputs(tests_path + "LinearAlgebra/svd-v.bin", outputs);
		run_generated_test<double>(inputs, outputs, [](const matrix::Matrix<double>& m1) { return std::get<2>(matrix::svd(m1)); });
	}
	TEST_METHOD(Rank) {
		std::vector<TestOutput<double>> outputs;
		read_outputs(tests_path + "LinearAlgebra/rank.bin", outputs);
		run_generated_test<double>(inputs, outputs, [](const matrix::Matrix<double>& m1) { return matrix::Matrix<double>(1, 1, { (double)matrix::rank(m1) }); });
	}
	TEST_METHOD(Determinant) {
		std::vector<TestOutput<double>> outputs;
		read_outputs(tests_path + "LinearAlgebra/determinant.bin", outputs);
		run_generated_test<double>(inputs, outputs, [](const matrix::Matrix<double>& m1) { return matrix::Matrix<double>(1, 1, { matrix::det(m1) }); });
	}
	TEST_METHOD(Trace) {
		std::vector<TestOutput<double>> outputs;
		read_outputs(tests_path + "LinearAlgebra/trace.bin", outputs);
		run_generated_test<double>(inputs, outputs, [](const matrix::Matrix<double>& m1) { return matrix::Matrix<double>(1, 1, { matrix::trace(m1) }); });
	}
	};
}