#include <iostream>

#include "../ndarray/matrix.hpp"
#include "../ndarray/linalg.hpp"
#include "../ndarray/ndarray.hpp"

#include "../lib_tests/generated_tests_io.hpp"

using namespace ndarray;


int main() {
    std::vector<ndarray::Matrix<double>> inputs;
    std::vector<generated_tests::TestOutput<double>> outputs;

    generated_tests::read_inputs(generated_tests::tests_path + "inputs.bin", inputs);
    generated_tests::read_outputs(generated_tests::tests_path + "LinearAlgebra/cholesky.bin", outputs);

    int i = 937;
    Matrix<double>& m1 = inputs[outputs[i].input_indices[0]];
    Matrix<double>& reference = outputs[i].output;

    Matrix<double> result = ndarray::cholesky(m1);

    Matrix<bool> eq = reference == result;
    Matrix<double> rel_err = (reference - result) / reference;

    double max_rel_err = 1e-6;
    std::cout << m1 << std::endl << reference << std::endl << result << std::endl << eq << std::endl << rel_err << std::endl << (rel_err.abs() < max_rel_err) << std::endl << ((rel_err.abs() < max_rel_err) || (reference == result)) << std::endl << ((rel_err.abs() < max_rel_err) || (reference == result)).reduce_all() << std::endl;

    return 0;
}