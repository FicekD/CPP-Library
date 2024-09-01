#ifndef _GENERAL_TESTS_H
#define _GENERAL_TESTS_H

#include <vector>

#include "generated_tests_io.hpp"
#include "CppUnitTest.h"
#include "../ndarray/matrix.hpp"

using namespace Microsoft::VisualStudio::CppUnitTestFramework;


template <typename T>
void run_generated_test(
    const std::vector<ndarray::Matrix<T>>& inputs,
    const std::vector<TestOutput<T>>& outputs,
    const std::function<ndarray::Matrix<T>(const ndarray::Matrix<T>&)>& operation)
{
    for (const TestOutput<T>& test : outputs) {
        const ndarray::Matrix<T>& input = inputs[test.input_indices[0]];
        const ndarray::Matrix<T>& result = operation(input);

        Assert::IsTrue(((result - test.output).abs() < 1e-30).reduce_all());
    }
}

template <typename T>
void run_generated_test(
    const std::vector<ndarray::Matrix<T>>& inputs,
    const std::vector<TestOutput<T>>& outputs,
    const std::function<ndarray::Matrix<T>(const ndarray::Matrix<T>&, const ndarray::Matrix<T>&)>& operation)
{
    for (const TestOutput<T>& test : outputs) {
        const ndarray::Matrix<T>& input_1 = inputs[test.input_indices[0]];
        const ndarray::Matrix<T>& input_2 = inputs[test.input_indices[1]];
        ndarray::Matrix<T> result = operation(input_1, input_2);

        Assert::IsTrue(((result - test.output).abs() < 1e-30).reduce_all());
    }
}

#endif
