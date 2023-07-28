#include "CppUnitTest.h"

#include "../ndarray/matrix.hpp"

using namespace Microsoft::VisualStudio::CppUnitTestFramework;

namespace arithmetic_ops_tests {
	TEST_CLASS(arithmetic_ops_tests) {
	public:
		TEST_METHOD(UnaryPlus) {}
		TEST_METHOD(UnaryMinus) {}
		TEST_METHOD(Increment) {}
		TEST_METHOD(Decrement) {}
		TEST_METHOD(Addition) {}
		TEST_METHOD(Subtraction) {}
		TEST_METHOD(Multiplication) {}
		TEST_METHOD(Division) {}
		TEST_METHOD(Square) {}
		TEST_METHOD(SquareRoot) {}
		TEST_METHOD(Power) {}
	};
}