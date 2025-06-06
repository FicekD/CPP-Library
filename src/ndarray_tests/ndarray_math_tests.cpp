#include <catch2/catch_test_macros.hpp>

#include "../ndarray/matrix.hpp"

#include "generated_tests_io.hpp"
#include "general_tests.hpp"

using namespace generated_tests;

namespace ndarray_math_tests {
	struct MatrixFixture {
		std::vector<ndarray::Matrix<double>> inputs;

		MatrixFixture() {
			read_inputs(tests_path + "inputs.bin", inputs);
		}
	};

	TEST_CASE_METHOD(MatrixFixture, "positive", "[ndarray][matrix][math]") {
		std::vector<TestOutput<double>> outputs;
		read_outputs(tests_path + "Arithmetic/positive.bin", outputs);
		run_generated_test<double>(inputs, outputs, [](const ndarray::Matrix<double>& m1) { return +m1; });
		run_generated_test<double>(inputs, outputs, [](const ndarray::Matrix<double>& m1) { return m1.positive(); });
		run_generated_test_inplace<double>(inputs, outputs, [](ndarray::Matrix<double>& m1) { m1.positive_inplace(); });
	}
	TEST_CASE_METHOD(MatrixFixture, "negative", "[ndarray][matrix][math]") {
		std::vector<TestOutput<double>> outputs;
		read_outputs(tests_path + "Arithmetic/negative.bin", outputs);
		run_generated_test<double>(inputs, outputs, [](const ndarray::Matrix<double>& m1) { return -m1; });
		run_generated_test<double>(inputs, outputs, [](const ndarray::Matrix<double>& m1) { return m1.negative(); });
		run_generated_test_inplace<double>(inputs, outputs, [](ndarray::Matrix<double>& m1) { m1.negative_inplace(); });
	}
	TEST_CASE_METHOD(MatrixFixture, "addition", "[ndarray][matrix][math]") {
		std::vector<TestOutput<double>> outputs;
		read_outputs(tests_path + "Arithmetic/addition.bin", outputs);
		run_generated_test<double>(inputs, outputs, [](const ndarray::Matrix<double>& m1, const ndarray::Matrix<double>& m2) { return m1 + m2; });
		run_generated_test<double>(inputs, outputs, [](const ndarray::Matrix<double>& m1, const ndarray::Matrix<double>& m2) { return m1.add(m2); });
		run_generated_test_inplace<double>(inputs, outputs, [](ndarray::Matrix<double>& m1, const ndarray::Matrix<double>& m2) { m1.add_inplace(m2); });
	}
	TEST_CASE_METHOD(MatrixFixture, "subtraction", "[ndarray][matrix][math]") {
		std::vector<TestOutput<double>> outputs;
		read_outputs(tests_path + "Arithmetic/subtraction.bin", outputs);
		run_generated_test<double>(inputs, outputs, [](const ndarray::Matrix<double>& m1, const ndarray::Matrix<double>& m2) { return m1 - m2; });
		run_generated_test<double>(inputs, outputs, [](const ndarray::Matrix<double>& m1, const ndarray::Matrix<double>& m2) { return m1.subtract(m2); });
		run_generated_test_inplace<double>(inputs, outputs, [](ndarray::Matrix<double>& m1, const ndarray::Matrix<double>& m2) { m1.subtract_inplace(m2); });
	}
	TEST_CASE_METHOD(MatrixFixture, "multiplication", "[ndarray][matrix][math]") {
		std::vector<TestOutput<double>> outputs;
		read_outputs(tests_path + "Arithmetic/multiplication.bin", outputs);
		run_generated_test<double>(inputs, outputs, [](const ndarray::Matrix<double>& m1, const ndarray::Matrix<double>& m2) { return m1 * m2; });
		run_generated_test<double>(inputs, outputs, [](const ndarray::Matrix<double>& m1, const ndarray::Matrix<double>& m2) { return m1.multiply(m2); });
		run_generated_test_inplace<double>(inputs, outputs, [](ndarray::Matrix<double>& m1, const ndarray::Matrix<double>& m2) { m1.multiply_inplace(m2); });
	}
	TEST_CASE_METHOD(MatrixFixture, "division", "[ndarray][matrix][math]") {
		std::vector<TestOutput<double>> outputs;
		read_outputs(tests_path + "Arithmetic/division.bin", outputs);
		run_generated_test<double>(inputs, outputs, [](const ndarray::Matrix<double>& m1, const ndarray::Matrix<double>& m2) { return m1 / m2; });
		run_generated_test<double>(inputs, outputs, [](const ndarray::Matrix<double>& m1, const ndarray::Matrix<double>& m2) { return m1.divide(m2); });
		run_generated_test_inplace<double>(inputs, outputs, [](ndarray::Matrix<double>& m1, const ndarray::Matrix<double>& m2) { m1.divide_inplace(m2); });
	}
	TEST_CASE_METHOD(MatrixFixture, "square", "[ndarray][matrix][math]") {
		std::vector<TestOutput<double>> outputs;
		read_outputs(tests_path + "Arithmetic/square.bin", outputs);
		run_generated_test<double>(inputs, outputs, [](const ndarray::Matrix<double>& m1) { return m1.square(); });
		run_generated_test_inplace<double>(inputs, outputs, [](ndarray::Matrix<double>& m1) { m1.square_inplace(); });
	}
	TEST_CASE_METHOD(MatrixFixture, "squareroot", "[ndarray][matrix][math]") {
		std::vector<TestOutput<double>> outputs;
		read_outputs(tests_path + "Arithmetic/squareroot.bin", outputs);
		run_generated_test<double>(inputs, outputs, [](const ndarray::Matrix<double>& m1) { return m1.sqrt(); });
		run_generated_test_inplace<double>(inputs, outputs, [](ndarray::Matrix<double>& m1) { m1.sqrt_inplace(); });
	}
	TEST_CASE_METHOD(MatrixFixture, "power", "[ndarray][matrix][math]") {
		std::vector<TestOutput<double>> outputs;
		read_outputs(tests_path + "Arithmetic/power.bin", outputs);
		run_generated_test<double>(inputs, outputs, [](const ndarray::Matrix<double>& m1, const ndarray::Matrix<double>& m2) { return m1.pow(m2); });
		run_generated_test_inplace<double>(inputs, outputs, [](ndarray::Matrix<double>& m1, const ndarray::Matrix<double>& m2) { m1.pow_inplace(m2); });
	}
	TEST_CASE_METHOD(MatrixFixture, "exp", "[ndarray][matrix][math]") {
		std::vector<TestOutput<double>> outputs;
		read_outputs(tests_path + "Arithmetic/exp.bin", outputs);
		run_generated_test<double>(inputs, outputs, [](const ndarray::Matrix<double>& m1) { return m1.exp(); });
		run_generated_test_inplace<double>(inputs, outputs, [](ndarray::Matrix<double>& m1) { m1.exp_inplace(); });
	}
	TEST_CASE_METHOD(MatrixFixture, "exp2", "[ndarray][matrix][math]") {
		std::vector<TestOutput<double>> outputs;
		read_outputs(tests_path + "Arithmetic/exp2.bin", outputs);
		run_generated_test<double>(inputs, outputs, [](const ndarray::Matrix<double>& m1) { return m1.exp2(); });
		run_generated_test_inplace<double>(inputs, outputs, [](ndarray::Matrix<double>& m1) { m1.exp2_inplace(); });
	}
	TEST_CASE_METHOD(MatrixFixture, "exp10", "[ndarray][matrix][math]") {
		std::vector<TestOutput<double>> outputs;
		read_outputs(tests_path + "Arithmetic/exp10.bin", outputs);
		run_generated_test<double>(inputs, outputs, [](const ndarray::Matrix<double>& m1) { return m1.exp10(); });
		run_generated_test_inplace<double>(inputs, outputs, [](ndarray::Matrix<double>& m1) { m1.exp10_inplace(); });
	}
	TEST_CASE_METHOD(MatrixFixture, "log", "[ndarray][matrix][math]") {
		std::vector<TestOutput<double>> outputs;
		read_outputs(tests_path + "Arithmetic/log.bin", outputs);
		run_generated_test<double>(inputs, outputs, [](const ndarray::Matrix<double>& m1) { return m1.log(); });
		run_generated_test_inplace<double>(inputs, outputs, [](ndarray::Matrix<double>& m1) { m1.log_inplace(); });
	}
	TEST_CASE_METHOD(MatrixFixture, "log2", "[ndarray][matrix][math]") {
		std::vector<TestOutput<double>> outputs;
		read_outputs(tests_path + "Arithmetic/log2.bin", outputs);
		run_generated_test<double>(inputs, outputs, [](const ndarray::Matrix<double>& m1) { return m1.log2(); });
		run_generated_test_inplace<double>(inputs, outputs, [](ndarray::Matrix<double>& m1) { m1.log2_inplace(); });
	}
	TEST_CASE_METHOD(MatrixFixture, "log10", "[ndarray][matrix][math]") {
		std::vector<TestOutput<double>> outputs;
		read_outputs(tests_path + "Arithmetic/log10.bin", outputs);
		run_generated_test<double>(inputs, outputs, [](const ndarray::Matrix<double>& m1) { return m1.log10(); });
		run_generated_test_inplace<double>(inputs, outputs, [](ndarray::Matrix<double>& m1) { m1.log10_inplace(); });
	}
	TEST_CASE_METHOD(MatrixFixture, "sin", "[ndarray][matrix][math]") {
		std::vector<TestOutput<double>> outputs;
		read_outputs(tests_path + "Arithmetic/sin.bin", outputs);
		run_generated_test<double>(inputs, outputs, [](const ndarray::Matrix<double>& m1) { return m1.sin(); });
		run_generated_test_inplace<double>(inputs, outputs, [](ndarray::Matrix<double>& m1) { m1.sin_inplace(); });
	}
	TEST_CASE_METHOD(MatrixFixture, "cos", "[ndarray][matrix][math]") {
		std::vector<TestOutput<double>> outputs;
		read_outputs(tests_path + "Arithmetic/cos.bin", outputs);
		run_generated_test<double>(inputs, outputs, [](const ndarray::Matrix<double>& m1) { return m1.cos(); });
		run_generated_test_inplace<double>(inputs, outputs, [](ndarray::Matrix<double>& m1) { m1.cos_inplace(); });
	}
	TEST_CASE_METHOD(MatrixFixture, "tan", "[ndarray][matrix][math]") {
		std::vector<TestOutput<double>> outputs;
		read_outputs(tests_path + "Arithmetic/tan.bin", outputs);
		run_generated_test<double>(inputs, outputs, [](const ndarray::Matrix<double>& m1) { return m1.tan(); });
		run_generated_test_inplace<double>(inputs, outputs, [](ndarray::Matrix<double>& m1) { m1.tan_inplace(); });
	}
	TEST_CASE_METHOD(MatrixFixture, "arcsin", "[ndarray][matrix][math]") {
		std::vector<TestOutput<double>> outputs;
		read_outputs(tests_path + "Arithmetic/arcsin.bin", outputs);
		run_generated_test<double>(inputs, outputs, [](const ndarray::Matrix<double>& m1) { return m1.arcsin(); });
		run_generated_test_inplace<double>(inputs, outputs, [](ndarray::Matrix<double>& m1) { m1.arcsin_inplace(); });
	}
	TEST_CASE_METHOD(MatrixFixture, "arccos", "[ndarray][matrix][math]") {
		std::vector<TestOutput<double>> outputs;
		read_outputs(tests_path + "Arithmetic/arccos.bin", outputs);
		run_generated_test<double>(inputs, outputs, [](const ndarray::Matrix<double>& m1) { return m1.arccos(); });
		run_generated_test_inplace<double>(inputs, outputs, [](ndarray::Matrix<double>& m1) { m1.arccos_inplace(); });
	}
	TEST_CASE_METHOD(MatrixFixture, "arctan", "[ndarray][matrix][math]") {
		std::vector<TestOutput<double>> outputs;
		read_outputs(tests_path + "Arithmetic/arctan.bin", outputs);
		run_generated_test<double>(inputs, outputs, [](const ndarray::Matrix<double>& m1) { return m1.arctan(); });
		run_generated_test_inplace<double>(inputs, outputs, [](ndarray::Matrix<double>& m1) { m1.arctan_inplace(); });
	}
	TEST_CASE_METHOD(MatrixFixture, "deg2rad", "[ndarray][matrix][math]") {
		std::vector<TestOutput<double>> outputs;
		read_outputs(tests_path + "Arithmetic/deg2rad.bin", outputs);
		run_generated_test<double>(inputs, outputs, [](const ndarray::Matrix<double>& m1) { return m1.deg2rad(); });
		run_generated_test_inplace<double>(inputs, outputs, [](ndarray::Matrix<double>& m1) { m1.deg2rad_inplace(); });
	}
	TEST_CASE_METHOD(MatrixFixture, "rad2deg", "[ndarray][matrix][math]") {
		std::vector<TestOutput<double>> outputs;
		read_outputs(tests_path + "Arithmetic/rad2deg.bin", outputs);
		run_generated_test<double>(inputs, outputs, [](const ndarray::Matrix<double>& m1) { return m1.rad2deg(); });
		run_generated_test_inplace<double>(inputs, outputs, [](ndarray::Matrix<double>& m1) { m1.rad2deg_inplace(); });
	}
	TEST_CASE_METHOD(MatrixFixture, "abs", "[ndarray][matrix][math]") {
		std::vector<TestOutput<double>> outputs;
		read_outputs(tests_path + "Arithmetic/abs.bin", outputs);
		run_generated_test<double>(inputs, outputs, [](const ndarray::Matrix<double>& m1) { return m1.abs(); });
		run_generated_test_inplace<double>(inputs, outputs, [](ndarray::Matrix<double>& m1) { m1.abs_inplace(); });
	}
	TEST_CASE_METHOD(MatrixFixture, "round", "[ndarray][matrix][math]") {
		std::vector<TestOutput<double>> outputs;
		read_outputs(tests_path + "Arithmetic/round.bin", outputs);
		run_generated_test<double>(inputs, outputs, [](const ndarray::Matrix<double>& m1) { return m1.round(); });
		run_generated_test_inplace<double>(inputs, outputs, [](ndarray::Matrix<double>& m1) { m1.round_inplace(); });
	}
	TEST_CASE_METHOD(MatrixFixture, "floor", "[ndarray][matrix][math]") {
		std::vector<TestOutput<double>> outputs;
		read_outputs(tests_path + "Arithmetic/floor.bin", outputs);
		run_generated_test<double>(inputs, outputs, [](const ndarray::Matrix<double>& m1) { return m1.floor(); });
		run_generated_test_inplace<double>(inputs, outputs, [](ndarray::Matrix<double>& m1) { m1.floor_inplace(); });
	}
	TEST_CASE_METHOD(MatrixFixture, "ceil", "[ndarray][matrix][math]") {
		std::vector<TestOutput<double>> outputs;
		read_outputs(tests_path + "Arithmetic/ceil.bin", outputs);
		run_generated_test<double>(inputs, outputs, [](const ndarray::Matrix<double>& m1) { return m1.ceil(); });
		run_generated_test_inplace<double>(inputs, outputs, [](ndarray::Matrix<double>& m1) { m1.ceil_inplace(); });
	}
	TEST_CASE_METHOD(MatrixFixture, "trunc", "[ndarray][matrix][math]") {
		std::vector<TestOutput<double>> outputs;
		read_outputs(tests_path + "Arithmetic/trunc.bin", outputs);
		run_generated_test<double>(inputs, outputs, [](const ndarray::Matrix<double>& m1) { return m1.trunc(); });
		run_generated_test_inplace<double>(inputs, outputs, [](ndarray::Matrix<double>& m1) { m1.trunc_inplace(); });
	}
	TEST_CASE_METHOD(MatrixFixture, "sign", "[ndarray][matrix][math]") {
		std::vector<TestOutput<double>> outputs;
		read_outputs(tests_path + "Arithmetic/sign.bin", outputs);
		run_generated_test<double>(inputs, outputs, [](const ndarray::Matrix<double>& m1) { return m1.sign(); });
		run_generated_test_inplace<double>(inputs, outputs, [](ndarray::Matrix<double>& m1) { m1.sign_inplace(); });
	}

	TEST_CASE_METHOD(MatrixFixture, "less", "[ndarray][matrix][math]") {
		std::vector<TestOutput<double>> outputs;
		read_outputs(tests_path + "Logical/less.bin", outputs);
		run_generated_test<double>(inputs, outputs, [](const ndarray::Matrix<double>& m1, const ndarray::Matrix<double>& m2) { return (m1 < m2).astype<double>(); });
		run_generated_test<double>(inputs, outputs, [](const ndarray::Matrix<double>& m1, const ndarray::Matrix<double>& m2) { return m1.less(m2).astype<double>(); });
		run_generated_test_inplace<double>(inputs, outputs, [](ndarray::Matrix<double>& m1, const ndarray::Matrix<double>& m2) { m1.less_inplace(m2); });
	}
	TEST_CASE_METHOD(MatrixFixture, "lessequal", "[ndarray][matrix][math]") {
		std::vector<TestOutput<double>> outputs;
		read_outputs(tests_path + "Logical/lessequal.bin", outputs);
		run_generated_test<double>(inputs, outputs, [](const ndarray::Matrix<double>& m1, const ndarray::Matrix<double>& m2) { return (m1 <= m2).astype<double>(); });
		run_generated_test<double>(inputs, outputs, [](const ndarray::Matrix<double>& m1, const ndarray::Matrix<double>& m2) { return m1.less_equal(m2).astype<double>(); });
		run_generated_test_inplace<double>(inputs, outputs, [](ndarray::Matrix<double>& m1, const ndarray::Matrix<double>& m2) { m1.less_equal_inplace(m2); });
	}
	TEST_CASE_METHOD(MatrixFixture, "greater", "[ndarray][matrix][math]") {
		std::vector<TestOutput<double>> outputs;
		read_outputs(tests_path + "Logical/greater.bin", outputs);
		run_generated_test<double>(inputs, outputs, [](const ndarray::Matrix<double>& m1, const ndarray::Matrix<double>& m2) { return (m1 > m2).astype<double>(); });
		run_generated_test<double>(inputs, outputs, [](const ndarray::Matrix<double>& m1, const ndarray::Matrix<double>& m2) { return m1.greater(m2).astype<double>(); });
		run_generated_test_inplace<double>(inputs, outputs, [](ndarray::Matrix<double>& m1, const ndarray::Matrix<double>& m2) { m1.greater_inplace(m2); });
	}
	TEST_CASE_METHOD(MatrixFixture, "greaterequal", "[ndarray][matrix][math]") {
		std::vector<TestOutput<double>> outputs;
		read_outputs(tests_path + "Logical/greaterequal.bin", outputs);
		run_generated_test<double>(inputs, outputs, [](const ndarray::Matrix<double>& m1, const ndarray::Matrix<double>& m2) { return (m1 >= m2).astype<double>(); });
		run_generated_test<double>(inputs, outputs, [](const ndarray::Matrix<double>& m1, const ndarray::Matrix<double>& m2) { return m1.greater_equal(m2).astype<double>(); });
		run_generated_test_inplace<double>(inputs, outputs, [](ndarray::Matrix<double>& m1, const ndarray::Matrix<double>& m2) { m1.greater_equal_inplace(m2); });
	}
	TEST_CASE_METHOD(MatrixFixture, "equal", "[ndarray][matrix][math]") {
		std::vector<TestOutput<double>> outputs;
		read_outputs(tests_path + "Logical/equal.bin", outputs);
		run_generated_test<double>(inputs, outputs, [](const ndarray::Matrix<double>& m1, const ndarray::Matrix<double>& m2) { return (m1 == m2).astype<double>(); });
		run_generated_test<double>(inputs, outputs, [](const ndarray::Matrix<double>& m1, const ndarray::Matrix<double>& m2) { return m1.equal(m2).astype<double>(); });
		run_generated_test_inplace<double>(inputs, outputs, [](ndarray::Matrix<double>& m1, const ndarray::Matrix<double>& m2) { m1.equal_inplace(m2); });
	}
	TEST_CASE_METHOD(MatrixFixture, "notequal", "[ndarray][matrix][math]") {
		std::vector<TestOutput<double>> outputs;
		read_outputs(tests_path + "Logical/notequal.bin", outputs);
		run_generated_test<double>(inputs, outputs, [](const ndarray::Matrix<double>& m1, const ndarray::Matrix<double>& m2) { return (m1 != m2).astype<double>(); });
		run_generated_test<double>(inputs, outputs, [](const ndarray::Matrix<double>& m1, const ndarray::Matrix<double>& m2) { return m1.not_equal(m2).astype<double>(); });
		run_generated_test_inplace<double>(inputs, outputs, [](ndarray::Matrix<double>& m1, const ndarray::Matrix<double>& m2) { m1.not_equal_inplace(m2); });
	}
	TEST_CASE_METHOD(MatrixFixture, "or", "[ndarray][matrix][math]") {
		std::vector<TestOutput<double>> outputs;
		read_outputs(tests_path + "Logical/logicalor.bin", outputs);
		run_generated_test<double>(inputs, outputs, [](const ndarray::Matrix<double>& m1, const ndarray::Matrix<double>& m2) { return (m1 || m2).astype<double>(); });
		run_generated_test<double>(inputs, outputs, [](const ndarray::Matrix<double>& m1, const ndarray::Matrix<double>& m2) { return m1.logical_or(m2).astype<double>(); });
		run_generated_test_inplace<double>(inputs, outputs, [](ndarray::Matrix<double>& m1, const ndarray::Matrix<double>& m2) { m1.logical_or_inplace(m2); });
	}
	TEST_CASE_METHOD(MatrixFixture, "and", "[ndarray][matrix][math]") {
		std::vector<TestOutput<double>> outputs;
		read_outputs(tests_path + "Logical/logicaland.bin", outputs);
		run_generated_test<double>(inputs, outputs, [](const ndarray::Matrix<double>& m1, const ndarray::Matrix<double>& m2) { return (m1 && m2).astype<double>(); });
		run_generated_test<double>(inputs, outputs, [](const ndarray::Matrix<double>& m1, const ndarray::Matrix<double>& m2) { return m1.logical_and(m2).astype<double>(); });
		run_generated_test_inplace<double>(inputs, outputs, [](ndarray::Matrix<double>& m1, const ndarray::Matrix<double>& m2) { m1.logical_and_inplace(m2); });
	}
	TEST_CASE_METHOD(MatrixFixture, "not", "[ndarray][matrix][math]") {
		std::vector<TestOutput<double>> outputs;
		read_outputs(tests_path + "Logical/logicalnot.bin", outputs);
		run_generated_test<double>(inputs, outputs, [](const ndarray::Matrix<double>& m1) { return (!m1).astype<double>(); });
		run_generated_test<double>(inputs, outputs, [](const ndarray::Matrix<double>& m1) { return m1.logical_not().astype<double>(); });
		run_generated_test_inplace<double>(inputs, outputs, [](ndarray::Matrix<double>& m1) { m1.logical_not_inplace(); });
	}
}