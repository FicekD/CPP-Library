#include "CppUnitTest.h"

#include "../ndarray/matrix.hpp"

#include "generated_tests_io.hpp"
#include "general_tests.hpp"

using namespace Microsoft::VisualStudio::CppUnitTestFramework;
using namespace generated_tests;

namespace arithmetic_tests {
	std::vector<ndarray::Matrix<double>> inputs;

	TEST_CLASS(Arithmetic) {
	public:
		TEST_CLASS_INITIALIZE(ReadGeneratedTestDefinitions)
		{
			if (inputs.empty())
				read_inputs(tests_path + "inputs.bin", inputs);
		}
		TEST_METHOD(Positive) {
			std::vector<TestOutput<double>> outputs;
			read_outputs(tests_path + "Arithmetic/positive.bin", outputs);
			run_generated_test<double>(inputs, outputs, [](const ndarray::Matrix<double>& m1) { return +m1; });
			run_generated_test_inplace<double>(inputs, outputs, [](ndarray::Matrix<double>& m1) { m1.inplace_positive(); });
		}
		TEST_METHOD(Negative) {
			std::vector<TestOutput<double>> outputs;
			read_outputs(tests_path + "Arithmetic/negative.bin", outputs);
			run_generated_test<double>(inputs, outputs, [](const ndarray::Matrix<double>& m1) { return -m1; });
			run_generated_test_inplace<double>(inputs, outputs, [](ndarray::Matrix<double>& m1) { m1.inplace_negative(); });
		}
		TEST_METHOD(Addition) {
			std::vector<TestOutput<double>> outputs;
			read_outputs(tests_path + "Arithmetic/addition.bin", outputs);
			run_generated_test<double>(inputs, outputs, [](const ndarray::Matrix<double>& m1, const ndarray::Matrix<double>& m2) { return m1 + m2; });
			run_generated_test<double>(inputs, outputs, [](const ndarray::Matrix<double>& m1, const ndarray::Matrix<double>& m2) { return m1.add(m2); });
			run_generated_test_inplace<double>(inputs, outputs, [](ndarray::Matrix<double>& m1, const ndarray::Matrix<double>& m2) { m1.add_inplace(m2); });
		}
		TEST_METHOD(Subtraction) {
			std::vector<TestOutput<double>> outputs;
			read_outputs(tests_path + "Arithmetic/subtraction.bin", outputs);
			run_generated_test<double>(inputs, outputs, [](const ndarray::Matrix<double>& m1, const ndarray::Matrix<double>& m2) { return m1 - m2; });
			run_generated_test<double>(inputs, outputs, [](const ndarray::Matrix<double>& m1, const ndarray::Matrix<double>& m2) { return m1.subtract(m2); });
			run_generated_test_inplace<double>(inputs, outputs, [](ndarray::Matrix<double>& m1, const ndarray::Matrix<double>& m2) { m1.subtract_inplace(m2); });
		}
		TEST_METHOD(Multiplication) {
			std::vector<TestOutput<double>> outputs;
			read_outputs(tests_path + "Arithmetic/multiplication.bin", outputs);
			run_generated_test<double>(inputs, outputs, [](const ndarray::Matrix<double>& m1, const ndarray::Matrix<double>& m2) { return m1 * m2; });
			run_generated_test<double>(inputs, outputs, [](const ndarray::Matrix<double>& m1, const ndarray::Matrix<double>& m2) { return m1.multiply(m2); });
			run_generated_test_inplace<double>(inputs, outputs, [](ndarray::Matrix<double>& m1, const ndarray::Matrix<double>& m2) { m1.multiply_inplace(m2); });
		}
		TEST_METHOD(Division) {
			std::vector<TestOutput<double>> outputs;
			read_outputs(tests_path + "Arithmetic/division.bin", outputs);
			run_generated_test<double>(inputs, outputs, [](const ndarray::Matrix<double>& m1, const ndarray::Matrix<double>& m2) { return m1 / m2; });
			run_generated_test<double>(inputs, outputs, [](const ndarray::Matrix<double>& m1, const ndarray::Matrix<double>& m2) { return m1.divide(m2); });
			run_generated_test_inplace<double>(inputs, outputs, [](ndarray::Matrix<double>& m1, const ndarray::Matrix<double>& m2) { m1.divide_inplace(m2); });
		}
		TEST_METHOD(Square) {
			std::vector<TestOutput<double>> outputs;
			read_outputs(tests_path + "Arithmetic/square.bin", outputs);
			run_generated_test<double>(inputs, outputs, [](const ndarray::Matrix<double>& m1) { return m1.square(); });
			run_generated_test_inplace<double>(inputs, outputs, [](ndarray::Matrix<double>& m1) { m1.square_inplace(); });
		}
		TEST_METHOD(SquareRoot) {
			std::vector<TestOutput<double>> outputs;
			read_outputs(tests_path + "Arithmetic/squareroot.bin", outputs);
			run_generated_test<double>(inputs, outputs, [](const ndarray::Matrix<double>& m1) { return m1.sqrt(); });
			run_generated_test_inplace<double>(inputs, outputs, [](ndarray::Matrix<double>& m1) { m1.sqrt_inplace(); });
		}
		TEST_METHOD(Power) {
			std::vector<TestOutput<double>> outputs;
			read_outputs(tests_path + "Arithmetic/power.bin", outputs);
			run_generated_test<double>(inputs, outputs, [](const ndarray::Matrix<double>& m1, const ndarray::Matrix<double>& m2) { return m1.pow(m2); });
			run_generated_test_inplace<double>(inputs, outputs, [](ndarray::Matrix<double>& m1, const ndarray::Matrix<double>& m2) { m1.pow_inplace(m2); });
		}
		TEST_METHOD(Exp) {
			std::vector<TestOutput<double>> outputs;
			read_outputs(tests_path + "Arithmetic/exp.bin", outputs);
			run_generated_test<double>(inputs, outputs, [](const ndarray::Matrix<double>& m1) { return m1.exp(); });
			run_generated_test_inplace<double>(inputs, outputs, [](ndarray::Matrix<double>& m1) { m1.exp_inplace(); });
		}
		TEST_METHOD(Exp2) {
			std::vector<TestOutput<double>> outputs;
			read_outputs(tests_path + "Arithmetic/exp2.bin", outputs);
			run_generated_test<double>(inputs, outputs, [](const ndarray::Matrix<double>& m1) { return m1.exp2(); });
			run_generated_test_inplace<double>(inputs, outputs, [](ndarray::Matrix<double>& m1) { m1.exp2_inplace(); });
		}
		TEST_METHOD(Exp10) {
			std::vector<TestOutput<double>> outputs;
			read_outputs(tests_path + "Arithmetic/exp10.bin", outputs);
			run_generated_test<double>(inputs, outputs, [](const ndarray::Matrix<double>& m1) { return m1.exp10(); });
			run_generated_test_inplace<double>(inputs, outputs, [](ndarray::Matrix<double>& m1) { m1.exp10_inplace(); });
		}
		TEST_METHOD(Log) {
			std::vector<TestOutput<double>> outputs;
			read_outputs(tests_path + "Arithmetic/log.bin", outputs);
			run_generated_test<double>(inputs, outputs, [](const ndarray::Matrix<double>& m1) { return m1.log(); });
			run_generated_test_inplace<double>(inputs, outputs, [](ndarray::Matrix<double>& m1) { m1.log_inplace(); });
		}
		TEST_METHOD(Log2) {
			std::vector<TestOutput<double>> outputs;
			read_outputs(tests_path + "Arithmetic/log2.bin", outputs);
			run_generated_test<double>(inputs, outputs, [](const ndarray::Matrix<double>& m1) { return m1.log2(); });
			run_generated_test_inplace<double>(inputs, outputs, [](ndarray::Matrix<double>& m1) { m1.log2_inplace(); });
		}
		TEST_METHOD(Log10) {
			std::vector<TestOutput<double>> outputs;
			read_outputs(tests_path + "Arithmetic/log10.bin", outputs);
			run_generated_test<double>(inputs, outputs, [](const ndarray::Matrix<double>& m1) { return m1.log10(); });
			run_generated_test_inplace<double>(inputs, outputs, [](ndarray::Matrix<double>& m1) { m1.log10_inplace(); });
		}
		TEST_METHOD(Sin) {
			std::vector<TestOutput<double>> outputs;
			read_outputs(tests_path + "Arithmetic/sin.bin", outputs);
			run_generated_test<double>(inputs, outputs, [](const ndarray::Matrix<double>& m1) { return m1.sin(); });
			run_generated_test_inplace<double>(inputs, outputs, [](ndarray::Matrix<double>& m1) { m1.sin_inplace(); });
		}
		TEST_METHOD(Cos) {
			std::vector<TestOutput<double>> outputs;
			read_outputs(tests_path + "Arithmetic/cos.bin", outputs);
			run_generated_test<double>(inputs, outputs, [](const ndarray::Matrix<double>& m1) { return m1.cos(); });
			run_generated_test_inplace<double>(inputs, outputs, [](ndarray::Matrix<double>& m1) { m1.cos_inplace(); });
		}
		TEST_METHOD(Tan) {
			std::vector<TestOutput<double>> outputs;
			read_outputs(tests_path + "Arithmetic/tan.bin", outputs);
			run_generated_test<double>(inputs, outputs, [](const ndarray::Matrix<double>& m1) { return m1.tan(); });
			run_generated_test_inplace<double>(inputs, outputs, [](ndarray::Matrix<double>& m1) { m1.tan_inplace(); });
		}
		TEST_METHOD(ArcSin) {
			std::vector<TestOutput<double>> outputs;
			read_outputs(tests_path + "Arithmetic/arcsin.bin", outputs);
			run_generated_test<double>(inputs, outputs, [](const ndarray::Matrix<double>& m1) { return m1.arcsin(); });
			run_generated_test_inplace<double>(inputs, outputs, [](ndarray::Matrix<double>& m1) { m1.arcsin_inplace(); });
		}
		TEST_METHOD(ArcCos) {
			std::vector<TestOutput<double>> outputs;
			read_outputs(tests_path + "Arithmetic/arccos.bin", outputs);
			run_generated_test<double>(inputs, outputs, [](const ndarray::Matrix<double>& m1) { return m1.arccos(); });
			run_generated_test_inplace<double>(inputs, outputs, [](ndarray::Matrix<double>& m1) { m1.arccos_inplace(); });
		}
		TEST_METHOD(ArcTan) {
			std::vector<TestOutput<double>> outputs;
			read_outputs(tests_path + "Arithmetic/arctan.bin", outputs);
			run_generated_test<double>(inputs, outputs, [](const ndarray::Matrix<double>& m1) { return m1.arctan(); });
			run_generated_test_inplace<double>(inputs, outputs, [](ndarray::Matrix<double>& m1) { m1.arctan_inplace(); });
		}
		TEST_METHOD(Deg2Rad) {
			std::vector<TestOutput<double>> outputs;
			read_outputs(tests_path + "Arithmetic/deg2rad.bin", outputs);
			run_generated_test<double>(inputs, outputs, [](const ndarray::Matrix<double>& m1) { return m1.deg2rad(); });
			run_generated_test_inplace<double>(inputs, outputs, [](ndarray::Matrix<double>& m1) { m1.deg2rad_inplace(); });
		}
		TEST_METHOD(Rad2Deg) {
			std::vector<TestOutput<double>> outputs;
			read_outputs(tests_path + "Arithmetic/rad2deg.bin", outputs);
			run_generated_test<double>(inputs, outputs, [](const ndarray::Matrix<double>& m1) { return m1.rad2deg(); });
			run_generated_test_inplace<double>(inputs, outputs, [](ndarray::Matrix<double>& m1) { m1.rad2deg_inplace(); });
		}
		TEST_METHOD(Abs) {
			std::vector<TestOutput<double>> outputs;
			read_outputs(tests_path + "Arithmetic/abs.bin", outputs);
			run_generated_test<double>(inputs, outputs, [](const ndarray::Matrix<double>& m1) { return m1.abs(); });
			run_generated_test_inplace<double>(inputs, outputs, [](ndarray::Matrix<double>& m1) { m1.abs_inplace(); });
		}
		TEST_METHOD(Round) {
			std::vector<TestOutput<double>> outputs;
			read_outputs(tests_path + "Arithmetic/round.bin", outputs);
			run_generated_test<double>(inputs, outputs, [](const ndarray::Matrix<double>& m1) { return m1.round(); });
			run_generated_test_inplace<double>(inputs, outputs, [](ndarray::Matrix<double>& m1) { m1.round_inplace(); });
		}
		TEST_METHOD(Floor) {
			std::vector<TestOutput<double>> outputs;
			read_outputs(tests_path + "Arithmetic/floor.bin", outputs);
			run_generated_test<double>(inputs, outputs, [](const ndarray::Matrix<double>& m1) { return m1.floor(); });
			run_generated_test_inplace<double>(inputs, outputs, [](ndarray::Matrix<double>& m1) { m1.floor_inplace(); });
		}
		TEST_METHOD(Ceil) {
			std::vector<TestOutput<double>> outputs;
			read_outputs(tests_path + "Arithmetic/ceil.bin", outputs);
			run_generated_test<double>(inputs, outputs, [](const ndarray::Matrix<double>& m1) { return m1.ceil(); });
			run_generated_test_inplace<double>(inputs, outputs, [](ndarray::Matrix<double>& m1) { m1.ceil_inplace(); });
		}
		TEST_METHOD(Trunc) {
			std::vector<TestOutput<double>> outputs;
			read_outputs(tests_path + "Arithmetic/trunc.bin", outputs);
			run_generated_test<double>(inputs, outputs, [](const ndarray::Matrix<double>& m1) { return m1.trunc(); });
			run_generated_test_inplace<double>(inputs, outputs, [](ndarray::Matrix<double>& m1) { m1.trunc_inplace(); });
		}
		TEST_METHOD(Sign) {
			std::vector<TestOutput<double>> outputs;
			read_outputs(tests_path + "Arithmetic/sign.bin", outputs);
			run_generated_test<double>(inputs, outputs, [](const ndarray::Matrix<double>& m1) { return m1.sign(); });
			run_generated_test_inplace<double>(inputs, outputs, [](ndarray::Matrix<double>& m1) { m1.sign_inplace(); });
		}
	};
}