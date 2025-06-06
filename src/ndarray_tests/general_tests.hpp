#ifndef _GENERAL_TESTS_H
#define _GENERAL_TESTS_H

#include <catch2/catch_test_macros.hpp>

#include <vector>

#include "generated_tests_io.hpp"
#include "../ndarray/matrix.hpp"


namespace generated_tests {
    constexpr double max_delta_abs = 1e-30;
    constexpr double max_delta_rel = 1e-6;

    template <typename T>
    static bool eval_results_abs_error(const ndarray::Matrix<T>& expected, const ndarray::Matrix<T>& output) {
        ndarray::Matrix<bool> eq = expected == output;
        if (eq.reduce_all())
            return true;
        ndarray::Matrix<T> abs_error = (expected - output).abs();
        return (eq || (abs_error < max_delta_abs)).reduce_all();
    }

    template <typename T>
    static bool eval_results_rel_error(const ndarray::Matrix<T>& expected, const ndarray::Matrix<T>& output) {
        ndarray::Matrix<bool> eq = expected == output;
        if (eq.reduce_all())
            return true;
        ndarray::Matrix<T> rel_error = (expected - output).abs() / expected;
        return (eq || (rel_error < max_delta_rel)).reduce_all();
    }

    static std::string get_err_msg(int i) {
        std::string s = "Failed on test case " + std::to_string(i);
        return s;
    }

    template <typename T>
    void run_generated_test(
        const std::vector<ndarray::Matrix<T>>& inputs,
        const std::vector<TestOutput<T>>& outputs,
        const std::function<ndarray::Matrix<T>(const ndarray::Matrix<T>&)>& operation,
        bool use_relative_erorr = false)
    {
        for (int i = 0; i < outputs.size(); i++) {
            const ndarray::Matrix<T>& input = inputs[outputs[i].input_indices[0]];
            const ndarray::Matrix<T>& result = operation(input);

            INFO(get_err_msg(i));
            if (use_relative_erorr)
                REQUIRE(eval_results_rel_error(result, outputs[i].output));
            else
                REQUIRE(eval_results_abs_error(result, outputs[i].output));
        }
    }

    template <typename T>
    void run_generated_test(
        const std::vector<ndarray::Matrix<T>>& inputs,
        const std::vector<TestOutput<T>>& outputs,
        const std::function<ndarray::Matrix<T>(const ndarray::Matrix<T>&, const ndarray::Matrix<T>&)>& operation,
        bool use_relative_erorr = false)
    {
        for (int i = 0; i < outputs.size(); i++) {
            const ndarray::Matrix<T>& input_1 = inputs[outputs[i].input_indices[0]];
            const ndarray::Matrix<T>& input_2 = inputs[outputs[i].input_indices[1]];
            ndarray::Matrix<T> result = operation(input_1, input_2);

            INFO(get_err_msg(i));
            if (use_relative_erorr)
                REQUIRE(eval_results_rel_error(result, outputs[i].output));
            else
                REQUIRE(eval_results_abs_error(result, outputs[i].output));
        }
    }

    template <typename T>
    void run_generated_test_inplace(
        const std::vector<ndarray::Matrix<T>>& inputs,
        const std::vector<TestOutput<T>>& outputs,
        const std::function<void(ndarray::Matrix<T>&)>& operation,
        bool use_relative_erorr = false)
    {
        for (int i = 0; i < outputs.size(); i++) {
            const ndarray::Matrix<T>& input = inputs[outputs[i].input_indices[0]];
            ndarray::Matrix<T> mutable_input(input);
            operation(mutable_input);

            INFO(get_err_msg(i));
            if (use_relative_erorr)
                REQUIRE(eval_results_rel_error(mutable_input, outputs[i].output));
            else
                REQUIRE(eval_results_abs_error(mutable_input, outputs[i].output));
        }
    }

    template <typename T>
    void run_generated_test_inplace(
        const std::vector<ndarray::Matrix<T>>& inputs,
        const std::vector<TestOutput<T>>& outputs,
        const std::function<void(ndarray::Matrix<T>&, const ndarray::Matrix<T>&)>& operation,
        bool use_relative_erorr = false)
    {
        for (int i = 0; i < outputs.size(); i++) {
            const ndarray::Matrix<T>& input_1 = inputs[outputs[i].input_indices[0]];
            ndarray::Matrix<T> input_1_mutable(input_1);
            const ndarray::Matrix<T>& input_2 = inputs[outputs[i].input_indices[1]];
            operation(input_1_mutable, input_2);

            INFO(get_err_msg(i));
            if (use_relative_erorr)
                REQUIRE(eval_results_rel_error(input_1_mutable, outputs[i].output));
            else
                REQUIRE(eval_results_abs_error(input_1_mutable, outputs[i].output));
        }
    }
}

#endif
