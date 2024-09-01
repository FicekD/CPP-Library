#include "CppUnitTest.h"

#include "../ndarray/matrix.hpp"

#include "generated_tests_io.hpp"
#include "general_tests.hpp"

using namespace Microsoft::VisualStudio::CppUnitTestFramework;


namespace arithmetic_ops_tests {
	std::vector<ndarray::Matrix<double>> inputs;

	TEST_CLASS(arithmetic_ops_tests) {
	public:
		TEST_CLASS_INITIALIZE(ReadGeneratedTestDefinitions)
		{
			if (inputs.empty())
				read_inputs(tests_path + "inputs.bin", inputs);
		}
		TEST_METHOD(UnaryPlus) {
			std::vector<TestOutput<double>> outputs;
			read_outputs(tests_path + "Arithmetic/positive.bin", outputs);
			run_generated_test<double>(inputs, outputs, [](const ndarray::Matrix<double> m1) { return +m1; });
		}
		TEST_METHOD(UnaryMinus) {
			std::vector<TestOutput<double>> outputs;
			read_outputs(tests_path + "Arithmetic/negative.bin", outputs);
			run_generated_test<double>(inputs, outputs, [](const ndarray::Matrix<double> m1) { return -m1; });
		}
		TEST_METHOD(Addition) {
			std::vector<TestOutput<double>> outputs;
			read_outputs(tests_path + "Arithmetic/addition.bin", outputs);
			run_generated_test<double>(inputs, outputs, [](const ndarray::Matrix<double> m1, const ndarray::Matrix<double> m2) { return m1 + m2; });
		}
		TEST_METHOD(Subtraction) {
			std::vector<TestOutput<double>> outputs;
			read_outputs(tests_path + "Arithmetic/subtraction.bin", outputs);
			run_generated_test<double>(inputs, outputs, [](const ndarray::Matrix<double> m1, const ndarray::Matrix<double> m2) { return m1 - m2; });
		}
		TEST_METHOD(Multiplication) {
			std::vector<TestOutput<double>> outputs;
			read_outputs(tests_path + "Arithmetic/multiplication.bin", outputs);
			run_generated_test<double>(inputs, outputs, [](const ndarray::Matrix<double> m1, const ndarray::Matrix<double> m2) { return m1 * m2; });
		}
		TEST_METHOD(Division) {
			std::vector<TestOutput<double>> outputs;
			read_outputs(tests_path + "Arithmetic/division.bin", outputs);
			run_generated_test<double>(inputs, outputs, [](const ndarray::Matrix<double> m1, const ndarray::Matrix<double> m2) { return m1 / m2; });
		}
		TEST_METHOD(Square) {
			std::vector<TestOutput<double>> outputs;
			read_outputs(tests_path + "Arithmetic/square.bin", outputs);
			run_generated_test<double>(inputs, outputs, [](const ndarray::Matrix<double> m1) { return m1.square(); });
		}
		TEST_METHOD(SquareRoot) {
			std::vector<TestOutput<double>> outputs;
			read_outputs(tests_path + "Arithmetic/squareroot.bin", outputs);
			run_generated_test<double>(inputs, outputs, [](const ndarray::Matrix<double> m1) { return m1.sqrt(); });
		}
		TEST_METHOD(Power) {
			std::vector<TestOutput<double>> outputs;
			read_outputs(tests_path + "Arithmetic/power.bin", outputs);
			run_generated_test<double>(inputs, outputs, [](const ndarray::Matrix<double> m1, const ndarray::Matrix<double> m2) { return m1.pow(m2); });
		}
		TEST_METHOD(Exp) {
			std::vector<TestOutput<double>> outputs;
			read_outputs(tests_path + "Arithmetic/exp.bin", outputs);
			run_generated_test<double>(inputs, outputs, [](const ndarray::Matrix<double> m1) { return m1.exp(); });
		}
		TEST_METHOD(Exp2) {
			std::vector<TestOutput<double>> outputs;
			read_outputs(tests_path + "Arithmetic/exp2.bin", outputs);
			run_generated_test<double>(inputs, outputs, [](const ndarray::Matrix<double> m1) { return m1.exp2(); });
		}
		TEST_METHOD(Exp10) {
			std::vector<TestOutput<double>> outputs;
			read_outputs(tests_path + "Arithmetic/exp10.bin", outputs);
			run_generated_test<double>(inputs, outputs, [](const ndarray::Matrix<double> m1) { return m1.exp10(); });
		}
		TEST_METHOD(Log) {
			std::vector<TestOutput<double>> outputs;
			read_outputs(tests_path + "Arithmetic/log.bin", outputs);
			run_generated_test<double>(inputs, outputs, [](const ndarray::Matrix<double> m1) { return m1.log(); });
		}
		TEST_METHOD(Log2) {
			std::vector<TestOutput<double>> outputs;
			read_outputs(tests_path + "Arithmetic/log2.bin", outputs);
			run_generated_test<double>(inputs, outputs, [](const ndarray::Matrix<double> m1) { return m1.log2(); });
		}
		TEST_METHOD(Log10) {
			std::vector<TestOutput<double>> outputs;
			read_outputs(tests_path + "Arithmetic/log10.bin", outputs);
			run_generated_test<double>(inputs, outputs, [](const ndarray::Matrix<double> m1) { return m1.log10(); });
		}
		TEST_METHOD(Sin) {
			std::vector<TestOutput<double>> outputs;
			read_outputs(tests_path + "Arithmetic/sin.bin", outputs);
			run_generated_test<double>(inputs, outputs, [](const ndarray::Matrix<double> m1) { return m1.sin(); });
		}
		TEST_METHOD(Cos) {
			std::vector<TestOutput<double>> outputs;
			read_outputs(tests_path + "Arithmetic/cos.bin", outputs);
			run_generated_test<double>(inputs, outputs, [](const ndarray::Matrix<double> m1) { return m1.cos(); });
		}
		TEST_METHOD(Tan) {
			std::vector<TestOutput<double>> outputs;
			read_outputs(tests_path + "Arithmetic/tan.bin", outputs);
			run_generated_test<double>(inputs, outputs, [](const ndarray::Matrix<double> m1) { return m1.tan(); });
		}
		TEST_METHOD(ArcSin) {
			std::vector<TestOutput<double>> outputs;
			read_outputs(tests_path + "Arithmetic/arcsin.bin", outputs);
			run_generated_test<double>(inputs, outputs, [](const ndarray::Matrix<double> m1) { return m1.arcsin(); });
		}
		TEST_METHOD(ArcCos) {
			std::vector<TestOutput<double>> outputs;
			read_outputs(tests_path + "Arithmetic/arccos.bin", outputs);
			run_generated_test<double>(inputs, outputs, [](const ndarray::Matrix<double> m1) { return m1.arccos(); });
		}
		TEST_METHOD(ArcTan) {
			std::vector<TestOutput<double>> outputs;
			read_outputs(tests_path + "Arithmetic/arctan.bin", outputs);
			run_generated_test<double>(inputs, outputs, [](const ndarray::Matrix<double> m1) { return m1.arctan(); });
		}
		TEST_METHOD(Deg2Rad) {
			std::vector<TestOutput<double>> outputs;
			read_outputs(tests_path + "Arithmetic/deg2rad.bin", outputs);
			run_generated_test<double>(inputs, outputs, [](const ndarray::Matrix<double> m1) { return m1.deg2rad(); });
		}
		TEST_METHOD(Rad2Deg) {
			std::vector<TestOutput<double>> outputs;
			read_outputs(tests_path + "Arithmetic/rad2deg.bin", outputs);
			run_generated_test<double>(inputs, outputs, [](const ndarray::Matrix<double> m1) { return m1.rad2deg(); });
		}
		TEST_METHOD(Abs) {
			std::vector<TestOutput<double>> outputs;
			read_outputs(tests_path + "Arithmetic/abs.bin", outputs);
			run_generated_test<double>(inputs, outputs, [](const ndarray::Matrix<double> m1) { return m1.abs(); });
		}
		TEST_METHOD(Round) {
			std::vector<TestOutput<double>> outputs;
			read_outputs(tests_path + "Arithmetic/round.bin", outputs);
			run_generated_test<double>(inputs, outputs, [](const ndarray::Matrix<double> m1) { return m1.round(); });
		}
		TEST_METHOD(Floor) {
			std::vector<TestOutput<double>> outputs;
			read_outputs(tests_path + "Arithmetic/floor.bin", outputs);
			run_generated_test<double>(inputs, outputs, [](const ndarray::Matrix<double> m1) { return m1.floor(); });
		}
		TEST_METHOD(Ceil) {
			std::vector<TestOutput<double>> outputs;
			read_outputs(tests_path + "Arithmetic/ceil.bin", outputs);
			run_generated_test<double>(inputs, outputs, [](const ndarray::Matrix<double> m1) { return m1.ceil(); });
		}
		TEST_METHOD(Trunc) {
			std::vector<TestOutput<double>> outputs;
			read_outputs(tests_path + "Arithmetic/trunc.bin", outputs);
			run_generated_test<double>(inputs, outputs, [](const ndarray::Matrix<double> m1) { return m1.trunc(); });
		}
		TEST_METHOD(Sign) {
			std::vector<TestOutput<double>> outputs;
			read_outputs(tests_path + "Arithmetic/sign.bin", outputs);
			run_generated_test<double>(inputs, outputs, [](const ndarray::Matrix<double> m1) { return m1.sign(); });
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