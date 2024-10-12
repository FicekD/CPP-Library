#include "CppUnitTest.h"

#include "../matrix/matrix.hpp"

#include "generated_tests_io.hpp"
#include "general_tests.hpp"

using namespace Microsoft::VisualStudio::CppUnitTestFramework;
using namespace generated_tests;

namespace bitwise_tests {
	std::vector<ndarray::Matrix<double>> inputs;

	TEST_CLASS(Bitwise) {
public:
	TEST_CLASS_INITIALIZE(ReadGeneratedTestDefinitions)
	{
		if (inputs.empty())
			read_inputs(tests_path + "inputs.bin", inputs);
	}
	};
}\