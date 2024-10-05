#include "CppUnitTest.h"

#include "../matrix/matrix.hpp"

#include "generated_tests_io.hpp"
#include "general_tests.hpp"

using namespace Microsoft::VisualStudio::CppUnitTestFramework;
using namespace generated_tests;

namespace logical_tests {
	std::vector<matrix::Matrix<double>> inputs;

	TEST_CLASS(Logical) {
public:
	TEST_CLASS_INITIALIZE(ReadGeneratedTestDefinitions)
	{
		if (inputs.empty())
			read_inputs(tests_path + "inputs.bin", inputs);
	}
	TEST_METHOD(Less) {
		std::vector<TestOutput<double>> outputs;
		read_outputs(tests_path + "Logical/less.bin", outputs);
		run_generated_test<double>(inputs, outputs, [](const matrix::Matrix<double>& m1, const matrix::Matrix<double>& m2) { return (m1 < m2).astype<double>(); });
		run_generated_test<double>(inputs, outputs, [](const matrix::Matrix<double>& m1, const matrix::Matrix<double>& m2) { return m1.less(m2).astype<double>(); });
		run_generated_test_inplace<double>(inputs, outputs, [](matrix::Matrix<double>& m1, const matrix::Matrix<double>& m2) { m1.less_inplace(m2); });
	}
	TEST_METHOD(LessEqual) {
		std::vector<TestOutput<double>> outputs;
		read_outputs(tests_path + "Logical/lessequal.bin", outputs);
		run_generated_test<double>(inputs, outputs, [](const matrix::Matrix<double>& m1, const matrix::Matrix<double>& m2) { return (m1 <= m2).astype<double>(); });
		run_generated_test<double>(inputs, outputs, [](const matrix::Matrix<double>& m1, const matrix::Matrix<double>& m2) { return m1.less_equal(m2).astype<double>(); });
		run_generated_test_inplace<double>(inputs, outputs, [](matrix::Matrix<double>& m1, const matrix::Matrix<double>& m2) { m1.less_equal_inplace(m2); });
	}
	TEST_METHOD(Greater) {
		std::vector<TestOutput<double>> outputs;
		read_outputs(tests_path + "Logical/greater.bin", outputs);
		run_generated_test<double>(inputs, outputs, [](const matrix::Matrix<double>& m1, const matrix::Matrix<double>& m2) { return (m1 > m2).astype<double>(); });
		run_generated_test<double>(inputs, outputs, [](const matrix::Matrix<double>& m1, const matrix::Matrix<double>& m2) { return m1.greater(m2).astype<double>(); });
		run_generated_test_inplace<double>(inputs, outputs, [](matrix::Matrix<double>& m1, const matrix::Matrix<double>& m2) { m1.greater_inplace(m2); });
	}
	TEST_METHOD(GreaterEqual) {
		std::vector<TestOutput<double>> outputs;
		read_outputs(tests_path + "Logical/greaterequal.bin", outputs);
		run_generated_test<double>(inputs, outputs, [](const matrix::Matrix<double>& m1, const matrix::Matrix<double>& m2) { return (m1 >= m2).astype<double>(); });
		run_generated_test<double>(inputs, outputs, [](const matrix::Matrix<double>& m1, const matrix::Matrix<double>& m2) { return m1.greater_equal(m2).astype<double>(); });
		run_generated_test_inplace<double>(inputs, outputs, [](matrix::Matrix<double>& m1, const matrix::Matrix<double>& m2) { m1.greater_equal_inplace(m2); });
	}
	TEST_METHOD(Equal) {
		std::vector<TestOutput<double>> outputs;
		read_outputs(tests_path + "Logical/equal.bin", outputs);
		run_generated_test<double>(inputs, outputs, [](const matrix::Matrix<double>& m1, const matrix::Matrix<double>& m2) { return (m1 == m2).astype<double>(); });
		run_generated_test<double>(inputs, outputs, [](const matrix::Matrix<double>& m1, const matrix::Matrix<double>& m2) { return m1.equal(m2).astype<double>(); });
		run_generated_test_inplace<double>(inputs, outputs, [](matrix::Matrix<double>& m1, const matrix::Matrix<double>& m2) { m1.equal_inplace(m2); });
	}
	TEST_METHOD(NotEqual) {
		std::vector<TestOutput<double>> outputs;
		read_outputs(tests_path + "Logical/notequal.bin", outputs);
		run_generated_test<double>(inputs, outputs, [](const matrix::Matrix<double>& m1, const matrix::Matrix<double>& m2) { return (m1 != m2).astype<double>(); });
		run_generated_test<double>(inputs, outputs, [](const matrix::Matrix<double>& m1, const matrix::Matrix<double>& m2) { return m1.not_equal(m2).astype<double>(); });
		run_generated_test_inplace<double>(inputs, outputs, [](matrix::Matrix<double>& m1, const matrix::Matrix<double>& m2) { m1.not_equal_inplace(m2); });
	}
	TEST_METHOD(Or) {
		std::vector<TestOutput<double>> outputs;
		read_outputs(tests_path + "Logical/logicalor.bin", outputs);
		run_generated_test<double>(inputs, outputs, [](const matrix::Matrix<double>& m1, const matrix::Matrix<double>& m2) { return (m1 || m2).astype<double>(); });
		run_generated_test<double>(inputs, outputs, [](const matrix::Matrix<double>& m1, const matrix::Matrix<double>& m2) { return m1.logical_or(m2).astype<double>(); });
		run_generated_test_inplace<double>(inputs, outputs, [](matrix::Matrix<double>& m1, const matrix::Matrix<double>& m2) { m1.logical_or_inplace(m2); });
	}
	TEST_METHOD(And) {
		std::vector<TestOutput<double>> outputs;
		read_outputs(tests_path + "Logical/logicaland.bin", outputs);
		run_generated_test<double>(inputs, outputs, [](const matrix::Matrix<double>& m1, const matrix::Matrix<double>& m2) { return (m1 && m2).astype<double>(); });
		run_generated_test<double>(inputs, outputs, [](const matrix::Matrix<double>& m1, const matrix::Matrix<double>& m2) { return m1.logical_and(m2).astype<double>(); });
		run_generated_test_inplace<double>(inputs, outputs, [](matrix::Matrix<double>& m1, const matrix::Matrix<double>& m2) { m1.logical_and_inplace(m2); });
	}
	TEST_METHOD(Not) {
		std::vector<TestOutput<double>> outputs;
		read_outputs(tests_path + "Logical/logicalnot.bin", outputs);
		run_generated_test<double>(inputs, outputs, [](const matrix::Matrix<double>& m1) { return (!m1).astype<double>(); });
		run_generated_test<double>(inputs, outputs, [](const matrix::Matrix<double>& m1) { return m1.logical_not().astype<double>(); });
		run_generated_test_inplace<double>(inputs, outputs, [](matrix::Matrix<double>& m1) { m1.logical_not_inplace(); });
	}
	};
}\