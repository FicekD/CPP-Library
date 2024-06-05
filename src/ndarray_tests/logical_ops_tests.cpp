#include "CppUnitTest.h"

#include "../ndarray/matrix.hpp"

using namespace Microsoft::VisualStudio::CppUnitTestFramework;

namespace logical_ops_tests {
	TEST_CLASS(logical_ops_tests) {
public:
	TEST_METHOD(Less) {
		ndarray::Matrix<int> mat1(3, 7);
		ndarray::Matrix<int> mat2(3, 7);

		mat1.fill(5);
		mat2.fill(-5);

		Assert::IsTrue((mat2 < mat1).reduce_all());
		Assert::IsFalse((mat1 < mat2).reduce_any());

		mat2.fill(5);

		Assert::IsFalse((mat1 < mat2).reduce_any());
		Assert::IsFalse((mat2 < mat1).reduce_any());
	}
	TEST_METHOD(LessEqual) {
		ndarray::Matrix<int> mat1(3, 7);
		ndarray::Matrix<int> mat2(3, 7);

		mat1.fill(5);
		mat2.fill(-5);

		Assert::IsTrue((mat2 <= mat1).reduce_all());
		Assert::IsFalse((mat1 <= mat2).reduce_any());

		mat2.fill(5);

		Assert::IsTrue((mat1 <= mat2).reduce_all());
		Assert::IsTrue((mat2 <= mat1).reduce_all());
	}
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