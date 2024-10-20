#include "CppUnitTest.h"

#include "../matrix/matrix.hpp"
#include "../matrix/linalg.hpp"

#include "generated_tests_io.hpp"
#include "general_tests.hpp"

using namespace Microsoft::VisualStudio::CppUnitTestFramework;
using namespace generated_tests;

namespace linalg_tests {
	std::vector<ndarray::Matrix<double>> inputs;

	TEST_CLASS(MatrixLinearAlgebra) {
public:
	TEST_CLASS_INITIALIZE(ReadGeneratedTestDefinitions)
	{
		if (inputs.empty())
			read_inputs(tests_path + "inputs.bin", inputs);
	}
	TEST_METHOD(Dot) {
		std::vector<TestOutput<double>> outputs;
		read_outputs(tests_path + "LinearAlgebra/dot.bin", outputs);
		run_generated_test<double>(inputs, outputs, [](const ndarray::Matrix<double>& m1, const ndarray::Matrix<double>& m2) { return ndarray::dot(m1, m2); });
	}
	TEST_METHOD(Transpose) {
		std::vector<TestOutput<double>> outputs;
		read_outputs(tests_path + "LinearAlgebra/transpose.bin", outputs);
		run_generated_test<double>(inputs, outputs, [](const ndarray::Matrix<double>& m1) { return ndarray::transpose(m1); });
	}
	TEST_METHOD(Inverse) {
		std::vector<TestOutput<double>> outputs;
		read_outputs(tests_path + "LinearAlgebra/inverse.bin", outputs);
		run_generated_test<double>(inputs, outputs, [](const ndarray::Matrix<double>& m1) { return ndarray::inverse(m1); });
	}
	TEST_METHOD(PseudoInverse) {
		std::vector<TestOutput<double>> outputs;
		read_outputs(tests_path + "LinearAlgebra/pseudoinverse.bin", outputs);
		run_generated_test<double>(inputs, outputs, [](const ndarray::Matrix<double>& m1) { return ndarray::pseudo_inverse(m1); });
	}
	TEST_METHOD(EigVals) {
		std::vector<TestOutput<double>> outputs;
		read_outputs(tests_path + "LinearAlgebra/eigenvalues.bin", outputs);
		run_generated_test<double>(inputs, outputs, [](const ndarray::Matrix<double>& m1) { return std::get<0>(ndarray::eig(m1)); });
	}
	TEST_METHOD(EigVectors) {
		std::vector<TestOutput<double>> outputs;
		read_outputs(tests_path + "LinearAlgebra/eigenvectors.bin", outputs);
		run_generated_test<double>(inputs, outputs, [](const ndarray::Matrix<double>& m1) { return std::get<1>(ndarray::eig(m1)); });
	}
	TEST_METHOD(Cholesky) {
		std::vector<TestOutput<double>> outputs;
		read_outputs(tests_path + "LinearAlgebra/cholesky.bin", outputs);
		run_generated_test<double>(inputs, outputs, [](const ndarray::Matrix<double>& m1) { return ndarray::cholesky(m1); });
	}
	TEST_METHOD(SVD_U) {
		std::vector<TestOutput<double>> outputs;
		read_outputs(tests_path + "LinearAlgebra/svd-u.bin", outputs);
		run_generated_test<double>(inputs, outputs, [](const ndarray::Matrix<double>& m1) { return std::get<0>(ndarray::svd(m1)); });
	}
	TEST_METHOD(SVD_Sigma) {
		std::vector<TestOutput<double>> outputs;
		read_outputs(tests_path + "LinearAlgebra/svd-sigma.bin", outputs);
		run_generated_test<double>(inputs, outputs, [](const ndarray::Matrix<double>& m1) { return std::get<1>(ndarray::svd(m1)); });
	}
	TEST_METHOD(SVD_V) {
		std::vector<TestOutput<double>> outputs;
		read_outputs(tests_path + "LinearAlgebra/svd-v.bin", outputs);
		run_generated_test<double>(inputs, outputs, [](const ndarray::Matrix<double>& m1) { return std::get<2>(ndarray::svd(m1)); });
	}
	TEST_METHOD(Rank) {
		std::vector<TestOutput<double>> outputs;
		read_outputs(tests_path + "LinearAlgebra/rank.bin", outputs);
		run_generated_test<double>(inputs, outputs, [](const ndarray::Matrix<double>& m1) { return ndarray::Matrix<double>(1, 1, { (double)ndarray::rank(m1) }); });
	}
	TEST_METHOD(Determinant) {
		std::vector<TestOutput<double>> outputs;
		read_outputs(tests_path + "LinearAlgebra/determinant.bin", outputs);
		run_generated_test<double>(inputs, outputs, [](const ndarray::Matrix<double>& m1) { return ndarray::Matrix<double>(1, 1, { ndarray::det(m1) }); });
	}
	TEST_METHOD(Trace) {
		std::vector<TestOutput<double>> outputs;
		read_outputs(tests_path + "LinearAlgebra/trace.bin", outputs);
		run_generated_test<double>(inputs, outputs, [](const ndarray::Matrix<double>& m1) { return ndarray::Matrix<double>(1, 1, { ndarray::trace(m1) }); });
	}
	};
}