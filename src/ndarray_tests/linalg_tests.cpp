#include "CppUnitTest.h"

#include "../ndarray/matrix.hpp"
#include "../ndarray/linalg.hpp"

using namespace Microsoft::VisualStudio::CppUnitTestFramework;

namespace linalg_tests {
	TEST_CLASS(linalg_tests) {
public:
	TEST_METHOD(Dot) {
		ndarray::Matrix<double> identity_3 = ndarray::Matrix<double>::eye(3, 0);
		ndarray::Matrix<double> identity_5 = ndarray::Matrix<double>::eye(5, 0);
		ndarray::Matrix<double> a(3, 5, std::vector<double> { 
			1.0, 2.0, 3.0, 4.0, 5.0,
			6.0, 7.0, 8.0, 9.0, 10.0,
			9.0, 8.0, 7.0, 6.0, 5.0,
		});
		ndarray::Matrix<double> b(5, 7, std::vector<double> {
			1.0, 2.0, 3.0, 4.0, 5.0, -1.0, -2.0,
			6.0, 7.0, 8.0, 9.0, 10.0, -1.0, -2.0,
			9.0, 8.0, 7.0, 6.0, 5.0, -1.0, -2.0,
			1.0, 2.0, 3.0, 4.0, 5.0, -1.0, -2.0,
			6.0, 7.0, 8.0, 9.0, 10.0, -1.0, -2.0,
		});

		ndarray::Matrix<double> r1 = ndarray::dot(identity_3, a);
		ndarray::Matrix<double> r2 = ndarray::dot(a, identity_5);

		Assert::IsTrue((r1 == a).reduce_all());
		Assert::IsTrue((r2 == a).reduce_all());
		Assert::IsTrue((r1 == r2).reduce_all());

		ndarray::Matrix<double> r3 = ndarray::dot(a, b);
		Assert::AreEqual(r3.rows(), a.rows());
		Assert::AreEqual(r3.cols(), b.cols());
	}
	TEST_METHOD(Transpose) {
		ndarray::Matrix<double> a(3, 5, std::vector<double> {
			1.0, 2.0, 3.0, 4.0, 5.0,
			6.0, 7.0, 8.0, 9.0, 10.0,
			9.0, 8.0, 7.0, 6.0, 5.0,
		});
		ndarray::Matrix<double> b = ndarray::transpose(a);

		Assert::AreEqual(a.rows(), b.cols());
		Assert::AreEqual(a.cols(), b.rows());

		for (size_t row = 0; row < a.rows(); row++) {
			for (size_t col = 0; col < a.cols(); col++) {
				Assert::AreEqual(a.get(row, col), b.get(col, row));
			}
		}
	}
	TEST_METHOD(Inverse) {
		ndarray::Matrix<double> identity = ndarray::Matrix<double>::eye(5, 0);
		ndarray::Matrix<double> inverted_identity = ndarray::inverse(identity);

		Assert::IsTrue((identity == inverted_identity).reduce_all());


	}
	TEST_METHOD(PseudoInverse) {
	
	}
	TEST_METHOD(Eig) {
	
	}
	TEST_METHOD(Cholesky) {
	
	}
	TEST_METHOD(SVD) {
	
	}
	TEST_METHOD(Rank) {
	
	}
	TEST_METHOD(Determinant) {
	
	}
	TEST_METHOD(Trace) {
	
	}
	};
}