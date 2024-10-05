#ifndef _GENERATED_TESTS_IO_H
#define _GENERATED_TESTS_IO_H

#include <string>
#include <fstream>

#include "../matrix/matrix.hpp"

namespace generated_tests {
    const std::string tests_path("D:/ALL-IN/Programming/C++/cpp-lib/data/mat_unit_tests/");

    template <typename T>
    struct TestOutput {
        std::vector<size_t> input_indices;
        matrix::Matrix<T> output;
    };

    template <typename T>
    static void bytes_to_value(char* bytes, T* val) {
        char* const p = reinterpret_cast<char*>(val);
        for (int i = 0; i < sizeof(T); ++i) {
            p[i] = bytes[i];
        }
    }

    template <typename T>
    matrix::Matrix<T> try_read_mat(std::ifstream& stream) {
        int width, height;
        char int_data[4];

        stream.read(int_data, 4);
        bytes_to_value(int_data, &width);

        if (stream.eof()) {
            throw std::ifstream::failure("");
        }

        stream.read(int_data, 4);
        bytes_to_value(int_data, &height);

        std::vector<char> bytes(width * height * sizeof(T));
        stream.read(bytes.data(), width * height * sizeof(T));
        double* const data = reinterpret_cast<T*>(bytes.data());
        matrix::Matrix<T> mat(height, width, data);

        return mat;
    }

    template <typename T>
    void read_inputs(const std::string& path, std::vector<matrix::Matrix<T>>& inputs) {
        std::ifstream stream(path, std::ios::binary);
        int i = 0;
        while (true) {
            i++;
            try {
                matrix::Matrix<T> mat = try_read_mat<T>(stream);
                inputs.push_back(mat);
            }
            catch (std::ifstream::failure) {
                break;
            }
        }

        stream.close();
    }

    template <typename T>
    void read_outputs(const std::string& path, std::vector<TestOutput<T>>& outputs) {
        std::ifstream stream(path, std::ios::binary);

        if (!stream.good()) {
            throw std::ifstream::failure("File not found");
        }

        char int_data[4];
        int n, index;
        while (true) {
            stream.read(int_data, 4);
            if (stream.eof()) {
                break;
            }
            bytes_to_value(int_data, &n);

            std::vector<size_t> indices(n);
            for (int i = 0; i < n; ++i) {
                stream.read(int_data, 4);
                bytes_to_value(int_data, &index);
                indices[i] = (size_t)index;
            }

            matrix::Matrix<T> mat = try_read_mat<T>(stream);

            TestOutput<T> test_output{ indices, mat };

            outputs.push_back(test_output);
        }

        stream.close();
    }
}

#endif