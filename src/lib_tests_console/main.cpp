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
    generated_tests::read_outputs(generated_tests::tests_path + "LinearAlgebra/trace.bin", outputs);

    int i = 20;
    Matrix<double>& m1 = inputs[outputs[i].input_indices[0]];
    double reference = outputs[i].output.get(0, 0);

    double result = ndarray::trace(m1);

    std::cout << m1 << std::endl << reference << std::endl << result << std::endl << (reference - result) << std::endl;

    return 0;
}