#ifndef _GENERAL_TESTS_H
#define _GENERAL_TESTS_H

#include <vector>

#include "generated_tests_io.hpp"
#include "CppUnitTest.h"
#include "../matrix/matrix.hpp"

using namespace Microsoft::VisualStudio::CppUnitTestFramework;


namespace generated_tests {
    constexpr double max_delta = 1e-30;

    template <typename T>
    bool eval_results(const matrix::Matrix<T>& m1, const matrix::Matrix<T>& m2) {
        if ((m1 == m2).reduce_all())
            return true;
        return (((m1 - m2).abs() < max_delta).reduce_all());
    }

    template <typename T>
    void run_generated_test(
        const std::vector<matrix::Matrix<T>>& inputs,
        const std::vector<TestOutput<T>>& outputs,
        const std::function<matrix::Matrix<T>(const matrix::Matrix<T>&)>& operation)
    {
        for (const TestOutput<T>& test : outputs) {
            const matrix::Matrix<T>& input = inputs[test.input_indices[0]];
            const matrix::Matrix<T>& result = operation(input);
            
            Assert::IsTrue(eval_results(result, test.output));
        }
    }

    template <typename T>
    void run_generated_test(
        const std::vector<matrix::Matrix<T>>& inputs,
        const std::vector<TestOutput<T>>& outputs,
        const std::function<matrix::Matrix<T>(const matrix::Matrix<T>&, const matrix::Matrix<T>&)>& operation)
    {
        for (const TestOutput<T>& test : outputs) {
            const matrix::Matrix<T>& input_1 = inputs[test.input_indices[0]];
            const matrix::Matrix<T>& input_2 = inputs[test.input_indices[1]];
            matrix::Matrix<T> result = operation(input_1, input_2);

            Assert::IsTrue(eval_results(result, test.output));
        }
    }

    template <typename T>
    void run_generated_test_inplace(
        const std::vector<matrix::Matrix<T>>& inputs,
        const std::vector<TestOutput<T>>& outputs,
        const std::function<void(matrix::Matrix<T>&)>& operation)
    {
        for (const TestOutput<T>& test : outputs) {
            const matrix::Matrix<T>& input = inputs[test.input_indices[0]];
            matrix::Matrix<T> mutable_input(input);
            operation(mutable_input);

            Assert::IsTrue(eval_results(mutable_input, test.output));
        }
    }

    template <typename T>
    void run_generated_test_inplace(
        const std::vector<matrix::Matrix<T>>& inputs,
        const std::vector<TestOutput<T>>& outputs,
        const std::function<void(matrix::Matrix<T>&, const matrix::Matrix<T>&)>& operation)
    {
        for (const TestOutput<T>& test : outputs) {
            const matrix::Matrix<T>& input_1 = inputs[test.input_indices[0]];
            matrix::Matrix<T> input_1_mutable(input_1);
            const matrix::Matrix<T>& input_2 = inputs[test.input_indices[1]];
            operation(input_1_mutable, input_2);

            Assert::IsTrue(eval_results(input_1_mutable, test.output));
        }
    }
}

#endif
