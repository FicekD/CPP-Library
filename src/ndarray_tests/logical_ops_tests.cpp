#include "CppUnitTest.h"

#include "../ndarray/matrix.hpp"

using namespace Microsoft::VisualStudio::CppUnitTestFramework;

namespace logical_ops_tests {
	TEST_CLASS(logical_ops_tests) {
public:
	TEST_METHOD(Less) {}
	TEST_METHOD(LessEqual) {}
	TEST_METHOD(Greater) {}
	TEST_METHOD(GreaterEqual) {}
	TEST_METHOD(Equal) {}
	TEST_METHOD(NotEqual) {}
	TEST_METHOD(Or) {}
	TEST_METHOD(And) {}
	TEST_METHOD(ExclusiveOr) {}
	TEST_METHOD(Not) {}
	};
}