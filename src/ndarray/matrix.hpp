#ifndef _MATRIX_H
#define _MATRIX_H

#include <iostream>
#include <vector>
#include <memory>
#include <cstring>
#include <cmath>
#include <numeric>
#include <functional>
#include <format>
#include <algorithm>

#include "array.hpp"

namespace ndarray {
    enum Dim {
        ROWS, COLS
    };

    template <typename T>
    class Matrix : public BaseArray<T> {
    private:
        std::size_t _rows = 0, _cols = 0;
    public:

#pragma region CONSTRUCTORS
        Matrix() {}
        Matrix(const Matrix<T>& matrix) : _rows(matrix._rows), _cols(matrix._cols), BaseArray<T>(matrix.size()) {
            if (size() > 0) {
                BaseArray<T>::_data = std::unique_ptr<T>(new T[matrix.size()]);
                std::memcpy(BaseArray<T>::_data.get(), matrix.BaseArray<T>::_data.get(), matrix.size() * sizeof(T));
            }
        }
        Matrix(Matrix<T>&& matrix) noexcept : _rows(matrix._rows), _cols(matrix._cols), BaseArray<T>() {
            BaseArray<T>::_size = matrix.size();
            BaseArray<T>::_data = std::move(matrix.BaseArray<T>::_data);
            matrix.clear();
        }
        Matrix(std::size_t rows, std::size_t cols) : _rows(rows), _cols(cols), BaseArray<T>(cols * rows) {
            if (size() > 0)
                BaseArray<T>::_data = std::unique_ptr<T>(new T[size()] { T() });
        }
        Matrix(std::size_t rows, std::size_t cols, T* data) : _rows(rows), _cols(cols), BaseArray<T>(cols * rows) {
            if (size() > 0) {
                BaseArray<T>::_data = std::unique_ptr<T>(new T[size()]);
                std::memcpy(BaseArray<T>::_data.get(), data, size() * sizeof(T));
            }
        }
        Matrix(std::size_t rows, std::size_t cols, const std::vector<T>& data) : _rows(rows), _cols(cols), BaseArray<T>(cols * rows) {
            if (size() > 0) {
                BaseArray<T>::_data = std::unique_ptr<T>(new T[size()]);
                std::memcpy(BaseArray<T>::_data.get(), data.data(), size() * sizeof(T));
            }
        }
        Matrix(std::vector<Matrix<T>> matrices, Dim concat_dim) {
            if (matrices.size() == 0)
                throw std::invalid_argument("Empty matrix vector to concat");
            
            if (concat_dim == ROWS) {
                _cols = matrices[0].cols();
                _rows = 0;
                for (const Matrix<T>& mat : matrices) {
                    if (mat.cols() != _cols)
                        throw std::invalid_argument("Matrix column missmatch");
                    _rows += mat.rows();
                }
                BaseArray<T>::_size = _rows * _cols;
                BaseArray<T>::_data = std::unique_ptr<T>(new T[size()]);
                std::size_t ptr = 0;
                for (const Matrix<T>& mat : matrices) {
                    std::memcpy(BaseArray<T>::_data.get() + ptr, mat.BaseArray<T>::_data.get(), mat.size() * sizeof(T));
                    ptr += mat.size();
                }
            }
            else {
                _cols = 0;
                _rows = matrices[0].rows();
                for (const Matrix<T>& mat : matrices) {
                    if (mat.rows() != _rows)
                        throw std::invalid_argument("Matrix row missmatch");
                    _cols += mat.cols();
                }
                BaseArray<T>::_size = _rows * _cols;
                BaseArray<T>::_data = std::unique_ptr<T>(new T[size()]);
                std::size_t ptr = 0;
                for (std::size_t row = 0; row < _rows; row++) {
                    for (const Matrix<T>& mat : matrices) {
                        std::memcpy(BaseArray<T>::_data.get() + ptr, mat.BaseArray<T>::_data.get() + row * mat._cols, mat._cols * sizeof(T));
                        ptr += mat._cols;
                    }
                }
            }
        }

        static Matrix<T> eye(std::size_t dim_size, int k = 0) {
            size_t abs_k = std::abs(k);
            if (abs_k > dim_size)
                throw std::invalid_argument("k out of eye matrix dimensions");
            T one = T(1);
            Matrix<T> ret = Matrix(dim_size, dim_size);
            if (k >= 0)
                for (size_t i = 0; i < dim_size - abs_k; i++) {
                    ret.at(i, i + abs_k) = one;
                }
            else
                for (size_t i = 0; i < dim_size - abs_k; i++) {
                    ret.at(i + abs_k, i) = one;
                }
            return ret;
        }

#pragma endregion CONSTRUCTORS

#pragma region CORE
        using BaseArray<T>::size;
        using BaseArray<T>::at;
        using BaseArray<T>::get;
        using BaseArray<T>::fill;

        std::size_t rows() const { return _rows; }
        std::size_t cols() const { return _cols; }

        void clear() {
            _rows = 0;
            _cols = 0;
            BaseArray<T>::clear();
        }

        template <typename T1, typename T2>
        static void _except_on_2d_mismatch(const Matrix<T1>& m1, const Matrix<T2>& m2) {
            if (m1._rows != m2._rows || m1._cols != m2._cols)
                throw std::invalid_argument("Matricies shape mismatch");
        }

        std::size_t ravel_indices(std::size_t row, std::size_t col) const {
            return row * _cols + col;
        }

        T get(std::size_t row, std::size_t col) const {
            if (row >= _rows || col >= _cols)
                throw std::invalid_argument("Indices out of bounds");
            return get(ravel_indices(row, col));
        }
        T& at(std::size_t row, std::size_t col) {
            if (row >= _rows || col >= _cols)
                throw std::invalid_argument("Indices out of bounds");
            return at(ravel_indices(row, col));
        }

        void reshape(std::size_t rows, std::size_t cols) {
            if (rows * cols != size())
                throw std::invalid_argument(std::format("Reshape missmatch [{}x{}] -> [{}x{}]", _rows, _cols, rows, cols));
            _rows = rows;
            _cols = cols;
        }
        
        template<typename U>
        Matrix<U> astype() const {
            Matrix<U> result(_rows, _cols);
            for (int i = 0; i < size(); i++)
                result.at(i) = static_cast<U>(this->get(i));
            return result;
        }

        Matrix<T> sub_matrix(std::size_t row_0, std::size_t row_1, std::size_t col_0, std::size_t col_1) const {
            if (row_1 < row_0 || col_1 < col_0)
                throw std::invalid_argument("Indices out of bounds");
            std::size_t sub_mat_rows = row_1 - row_0, sub_mat_cols = col_1 - col_0;
            Matrix<T> result(sub_mat_rows, sub_mat_cols);
            for (std::size_t row = 0; row < sub_mat_rows; row++)
                for (std::size_t col = 0; col < sub_mat_cols; col++)
                    result.at(row, col) = this->get(row_0 + row, col_0 + col);
            return result;
        }

#pragma endregion CORE

#pragma region MATH_OPS

        using BaseArray<T>::add_inplace;
        using BaseArray<T>::subtract_inplace;
        using BaseArray<T>::multiply_inplace;
        using BaseArray<T>::divide_inplace;
        using BaseArray<T>::square_inplace;
        using BaseArray<T>::sqrt_inplace;
        using BaseArray<T>::pow_inplace;
        using BaseArray<T>::exp_inplace;
        using BaseArray<T>::exp2_inplace;
        using BaseArray<T>::exp10_inplace;
        using BaseArray<T>::log_inplace;
        using BaseArray<T>::log2_inplace;
        using BaseArray<T>::log10_inplace;

        using BaseArray<T>::sin_inplace;
        using BaseArray<T>::cos_inplace;
        using BaseArray<T>::tan_inplace;
        using BaseArray<T>::arcsin_inplace;
        using BaseArray<T>::arccos_inplace;
        using BaseArray<T>::arctan_inplace;
        using BaseArray<T>::deg2rad_inplace;
        using BaseArray<T>::rad2deg_inplace;
        
        using BaseArray<T>::abs_inplace;
        using BaseArray<T>::clamp_inplace;
        using BaseArray<T>::round_inplace;
        using BaseArray<T>::floor_inplace;
        using BaseArray<T>::ceil_inplace;
        using BaseArray<T>::trunc_inplace;

        template <typename R = T>
        Matrix<R> map_to_new(const std::function<R(const T&, const T&)>& lambda, const Matrix<T>& matrix) const {
            Matrix::_except_on_2d_mismatch<T, T>(*this, matrix);
            Matrix<R> result(_rows, _cols);
            BaseArray<T>::template map_to<R>(lambda, dynamic_cast<const BaseArray<T>&>(matrix), dynamic_cast<BaseArray<R>&>(result));
            return result;
        }
        template <typename R = T>
        Matrix<R> map_to_new(const std::function<R(const T&)>& lambda) const {
            Matrix<R> result(_rows, _cols);
            BaseArray<T>::template map_to<R>(lambda, dynamic_cast<BaseArray<R>&>(result));
            return result;
        }
        void operator=(const Matrix<T>& matrix) {
            if (matrix.size() != size()) {
                BaseArray<T>::_data.release();
                BaseArray<T>::_data = std::unique_ptr<T>(new T[matrix.size()]);
            }
            std::memcpy(BaseArray<T>::_data.get(), matrix.BaseArray<T>::_data.get(), matrix.size() * sizeof(T));
            BaseArray<T>::_size = matrix.size();
            _rows = matrix._rows;
            _cols = matrix._cols;
        }
        Matrix<T> operator+(const Matrix<T>& matrix) const {
            return map_to_new<T>([](const T& x1, const T& x2) -> T { return x1 + x2; }, matrix);
        }
        Matrix<T> operator+(const T& scalar) const {
            return map_to_new<T>([scalar](const T& x1) -> T { return x1 + scalar; });
        }
        Matrix<T> operator+() const {
            Matrix<T> result(*this);
            return result;
        }
        Matrix<T> operator-(const Matrix<T>& matrix) const {
            return map_to_new<T>([](const T& x1, const T& x2) -> T { return x1 - x2; }, matrix);
        }
        Matrix<T> operator-(const T& scalar) const {
            return map_to_new<T>([scalar](const T& x1) -> T { return x1 - scalar; });
        }
        Matrix<T> operator-() const {
            return map_to_new<T>([](const T& x1) -> T { return -x1; });
        }
        Matrix<T> operator*(const Matrix<T>& matrix) const {
            return map_to_new<T>([](const T& x1, const T& x2) -> T { return x1 * x2; }, matrix);
        }
        Matrix<T> operator*(const T& scalar) const {
            return map_to_new<T>([scalar](const T& x1) -> T { return x1 * scalar; });
        }
        Matrix<T> operator/(const Matrix<T>& matrix) const {
            return map_to_new<T>([](const T& x1, const T& x2) -> T { return x1 / x2; }, matrix);
        }
        Matrix<T> operator/(const T& scalar) const {
            return map_to_new<T>([scalar](const T& x1) -> T { return x1 / scalar; });
        }
        Matrix<T> square() const {
            return map_to_new<T>([](const T& x1) -> T { return x1 * x1; });
        }
        Matrix<T> sqrt() const {
            return map_to_new<T>([](const T& x1) -> T { return (T)std::sqrt(x1); });
        }
        Matrix<T> pow(double power) const {
            return map_to_new<T>([power](const T& x1) -> T { return (T)std::pow(x1, power); });
        }
        Matrix<T> exp() const {
            return map_to_new<T>([](const T& x1) -> T { return std::exp(x1); });
        }
        Matrix<T> exp2() const {
            return map_to_new<T>([](const T& x1) -> T { return std::exp2(x1); });
        }
        Matrix<T> exp10() const {
            return map_to_new<T>([](const T& x1) -> T { return pow(T(10), x1); });
        }
        Matrix<T> log() const {
            return map_to_new<T>([](const T& x1) -> T { return std::log(x1); });
        }
        Matrix<T> log2() const {
            return map_to_new<T>([](const T& x1) -> T { return std::sqrt(x1); });
        }
        Matrix<T> log10() const {
            return map_to_new<T>([](const T& x1) -> T { return std::log10(x1); });
        }
        Matrix<T> sin() const {
            return map_to_new<T>([](const T& x1) -> T { return std::sin(x1); });
        }
        Matrix<T> cos() const {
            return map_to_new<T>([](const T& x1) -> T { return std::cos(x1); });
        }
        Matrix<T> tan() const {
            return map_to_new<T>([](const T& x1) -> T { return std::tan(x1); });
        }
        Matrix<T> arcsin() const {
            return map_to_new<T>([](const T& x1) -> T { return std::asin(x1); });
        }
        Matrix<T> arccos() const {
            return map_to_new<T>([](const T& x1) -> T { return std::acos(x1); });
        }
        Matrix<T> arctan() const {
            return map_to_new<T>([](const T& x1) -> T { return std::atan(x1); });
        }
        Matrix<T> deg2rad() const {
            return map_to_new<T>([](const T& x1) -> T { x1 * T(PI / 180.0); });
        }
        Matrix<T> rad2deg() const {
            return map_to_new<T>([](const T& x1) -> T { x1 * T(180.0 / PI); });
        }
        Matrix<T> abs() const {
            return map_to_new<T>([](const T& x1) -> T { return std::abs(x1); });
        }
        Matrix<int> sign() const {
            return map_to_new<T>([](const T& x1) -> T { (x1 > T(0)) - (x1 < T(0)); });
        }
        Matrix<T> clamp(const T& min, const T& max) const {
            return map_to_new<T>([min, max](const T& x1) -> T { return std::clamp(x1, min, max); });
        }
        Matrix<T> round() const {
            return map_to_new<T>([](const T& x1) -> T { return std::round(x1); });
        }
        Matrix<T> floor() const {
            return map_to_new<T>([](const T& x1) -> T { return std::floor(x1); });
        }
        Matrix<T> ceil() const {
            return map_to_new<T>([](const T& x1) -> T { return std::ceil(x1); });
        }
        Matrix<T> trunc() const {
            return map_to_new<T>([](const T& x1) -> T { return std::trunc(x1); });
        }

#pragma endregion MATH_OPS

#pragma region LOGICAL_OPS
        Matrix<bool> operator<(const Matrix<T>& matrix) const {
            return map_to_new<bool>([](const T& x1, const T& x2) -> bool { return x1 < x2; }, matrix);
        }
        Matrix<bool> operator<(const T& scalar) const {
            return map_to_new<bool>([scalar](const T& x1) -> bool { return x1 < scalar; });
        }
        Matrix<bool> operator<=(const Matrix<T>& matrix) const {
            return map_to_new<bool>([](const T& x1, const T& x2) -> bool { return x1 <= x2; }, matrix);
        }
        Matrix<bool> operator<=(const T& scalar) const {
            return map_to_new<bool>([scalar](const T& x1) -> bool { return x1 <= scalar; });
        }
        Matrix<bool> operator>(const Matrix<T>& matrix) const {
            return map_to_new<bool>([](const T& x1, const T& x2) -> bool { return x1 > x2; }, matrix);
        }
        Matrix<bool> operator>(const T& scalar) const {
            return map_to_new<bool>([scalar](const T& x1) -> bool { return x1 > scalar; });
        }
        Matrix<bool> operator>=(const Matrix<T>& matrix) const {
            return map_to_new<bool>([](const T& x1, const T& x2) -> bool { return x1 >= x2; }, matrix);
        }
        Matrix<bool> operator>=(const T& scalar) const {
            return map_to_new<bool>([scalar](const T& x1) -> bool { return x1 >= scalar; });
        }
        Matrix<bool> operator==(const Matrix<T>& matrix) const {
            return map_to_new<bool>([](const T& x1, const T& x2) -> bool { return x1 == x2; }, matrix);
        }
        Matrix<bool> operator==(const T& scalar) const {
            return map_to_new<bool>([scalar](const T& x1) -> bool { return x1 == scalar; });
        }
        Matrix<bool> operator!=(const Matrix<T>& matrix) const {
            return map_to_new<bool>([](const T& x1, const T& x2) -> bool { return x1 != x2; }, matrix);
        }
        Matrix<bool> operator!=(const T& scalar) const {
            return map_to_new<bool>([scalar](const T& x1) -> bool { return x1 != scalar; });
        }
        Matrix<bool> operator^(const Matrix<T>& matrix) const {
            return map_to_new<bool>([](const T& x1, const T& x2) -> bool { return x1 ^ x2; }, matrix);
        }
        Matrix<bool> operator^(const T& scalar) const {
            return map_to_new<bool>([scalar](const T& x1) -> bool { return x1 ^ scalar; });
        }
        Matrix<bool> operator||(const Matrix<T>& matrix) const {
            return map_to_new<bool>([](const T& x1, const T& x2) -> bool { return x1 || x2; }, matrix);
        }
        Matrix<bool> operator||(const T& scalar) const {
            return map_to_new<bool>([scalar](const T& x1) -> bool { return x1 || scalar; });
        }
        Matrix<bool> operator&&(const Matrix<T>& matrix) const {
            return map_to_new<bool>([](const T& x1, const T& x2) -> bool { return x1 && x2; }, matrix);
        }
        Matrix<bool> operator&&(const T& scalar) const {
            return map_to_new<bool>([scalar](const T& x1) -> bool { return x1 && scalar; });
        }
        Matrix<bool> operator!() const {
            return map_to_new<bool>([](const T& x1) -> bool { return !x1; });
        }
#pragma endregion LOGICAL_OPS
        
#pragma region REDUCE
        using BaseArray<T>::reduce_sum;
        using BaseArray<T>::reduce_prod;
        using BaseArray<T>::reduce_max;
        using BaseArray<T>::reduce_min;
        using BaseArray<T>::reduce_any;
        using BaseArray<T>::reduce_all;
        
        template<typename R>
        Matrix<R> reduce_rows(const std::function<R(const R&, const T&)>& lambda, const R& initializer) const {
            Matrix<R> result(1, _cols);
            result.fill(initializer);
            for (std::size_t row = 0; row < _rows; row++) {
                for (std::size_t col = 0; col < _cols; col++) {
                    R& at_col = result.BaseArray<R>::at(col);
                    at_col = lambda(at_col, this->get(row, col));
                }
            }
            return result;
        }
        template<typename R>
        Matrix<R> reduce_cols(const std::function<R(const R&, const T&)>& lambda, const R& initializer) const {
            Matrix<R> result(1, _cols);
            result.fill(initializer);
            for (std::size_t row = 0; row < _rows; row++) {
                for (std::size_t col = 0; col < _cols; col++) {
                    R& at_row = result.BaseArray<R>::at(row);
                    at_row = lambda(at_row, this->get(row, col));
                }
            }
            return result;
        }
        template<typename R>
        Matrix<R> reduce(const std::function<R(const R&, const T&)>& lambda, Dim dim, const R& initializer) const {
            if (dim == ROWS) {
                return reduce_rows<R>(lambda, initializer);
            }
            else {
                return reduce_cols<R>(lambda, initializer);
            }
        }
        Matrix<T> reduce_sum(Dim dim) const {
            return reduce<T>([](const T& x0, const T& x1) -> T { return x0 + x1; }, dim, T(0));
        }
        Matrix<T> reduce_prod(Dim dim) const {
            return reduce<T>([](const T& x0, const T& x1) -> T { return x0 * x1; }, dim, T(1));
        }
        Matrix<T> reduce_max(Dim dim) const {
            return reduce<T>([](const T& x0, const T& x1) -> T { return x0 > x1 ? x0 : x1; }, dim, T(-INFINITY));
        }
        Matrix<T> reduce_min(Dim dim) const {
            return reduce<T>([](const T& x0, const T& x1) -> T { return x0 < x1 ? x0 : x1; }, dim, T(INFINITY));
        }
        Matrix<bool> reduce_any(Dim dim) const {
            return reduce<bool>([](const T& x0, const T& x1) -> T { return x0 || x1 != 0; }, dim, false);
        }
        Matrix<bool> reduce_all(Dim dim) const {
            return reduce<bool>([](const T& x0, const T& x1) -> T { return x0 && x1 != 0; }, dim, true);
        }

#pragma endregion REDUCE

#pragma region LINALG
        Matrix<T> transpose() const {
            Matrix<T> mat(_cols, _rows);
            for (std::size_t row = 0; row < _rows; row++) {
                for (std::size_t col = 0; col < _cols; col++) {
                    mat.at(col, row) = this->at(row, col);
                }
            }
            return mat;
        }
        void transpose_inplace() {
            for (std::size_t row = 0; row < _rows; row++) {
                for (std::size_t col = 0; col < _cols; col++) {
                    if (col >= row)
                        break;
                    std::swap(this->at(ravel_indices(row, col)), this->at(ravel_indices(col, row)));
                }
            }
            this->reshape(_cols, _rows);
        }

        Matrix<T> dot(const Matrix<T>& matrix) const {
            if (this->_cols != matrix.rows())
                throw std::invalid_argument("Matricies shape missmatch");
            
            Matrix<T> result(_rows, matrix.cols());
            for (std::size_t row = 0; row < _rows; row++) {
                for (std::size_t col = 0; col < matrix.cols(); col++) {
                    T sum = T(0);
                    for (std::size_t i = 0; i < _cols; i++) {
                        sum += this->at(row, i) * matrix.at(i, col);
                    }
                    result.at(row, col) = sum;
                }
            }
            return result;
        }
        Matrix<T> inverse() const {
            if (_rows != _cols)
                throw std::invalid_argument("Matrix has to be square");
            
            Matrix<T> expanded(std::vector<Matrix<T>> { *this, Matrix<T>::eye(_rows) }, COLS);
        }
        Matrix<T> pseudo_inverse() const;

        std::tuple<Matrix<T>, Matrix<T>> eig() const;

        Matrix<T> cholesky() const;
        std::tuple<Matrix<T>, Matrix<T>, Matrix<T>> svd() const;

        std::size_t rank() const;
        T det() const;
        T trace(int k = 0) const;

#pragma endregion LINALG

#pragma region OTHER
        Matrix<T> flip_ud() const {
            Matrix<T> result(_rows, _cols);
            for (std::size_t row = 0; row < std::size_t(std::ceil(float(_rows) / 2)); row++) {
                for (std::size_t col = 0; col < _cols; col++) {
                    result.at(_rows - 1 - row, col) = this->get(row, col);
                    result.at(row, col) = this->get(_rows - 1 - row, col);
                }
            }
            return result;
        }
        Matrix<T> flip_lr() const {
            Matrix<T> result(_rows, _cols);
            for (std::size_t col = 0; col < std::size_t(std::ceil(float(_cols) / 2)); col++) {
                for (std::size_t row = 0; row < _rows; row++) {
                    result.at(row, _cols - 1 - col) = this->get(row, col);
                    result.at(row, col) = this->get(row, _cols - 1 - col);
                }
            }
            return result;
        }
        
        Matrix<T> pad(std::size_t pre_rows, std::size_t post_rows, std::size_t pre_cols, std::size_t post_cols, const T& scalar) const {
            Matrix<T> result(_rows + pre_rows + post_rows, _cols + pre_cols + post_cols);
            for (std::size_t row = 0; row < pre_rows; row++)
                for (std::size_t col = 0; col < result._cols; col++)
                    result.at(row, col) = scalar; 
            for (std::size_t row = pre_rows + _rows; row < result._rows; row++)
                for (std::size_t col = 0; col < result._cols; col++)
                    result.at(row, col) = scalar;
            for (std::size_t col = 0; col < pre_cols; col++)
                for (std::size_t row = pre_rows; row < result._rows - post_rows; row++)
                    result.at(row, col) = scalar;
            for (std::size_t col = pre_cols + _cols; col < result._cols; col++)
                for (std::size_t row = pre_rows; row < result._rows - post_rows; row++)
                    result.at(row, col) = scalar;
            
            for (std::size_t row = 0; row < _rows; row++)
                for (std::size_t col = 0; col < _cols; col++)
                    result.at(row + pre_rows, col + pre_cols) = this->get(row, col);
            return result;
        }

#pragma endregion OTHER

#pragma region ITERATOR
        class MatrixIterator {
        private:
            Matrix<T>* _mat_ptr;
            std::size_t _ptr;
            std::size_t _row_start, _col_start, _row_end, _col_end;
            std::size_t _iter_rows, _iter_cols, _iter_size;
            Dim _order;
        public:
            using difference_type = std::ptrdiff_t;
            using element_type = T;
            using pointer = element_type*;
            using reference = element_type&;

            MatrixIterator(Matrix<T>* mat_ptr, Dim order, std::size_t ptr,
                           std::size_t row_start, std::size_t row_end,
                           std::size_t col_start, std::size_t col_end)
                : _mat_ptr(mat_ptr), _ptr(ptr), _row_start(row_start), _col_start(col_start), _row_end(row_end), _col_end(col_end), _order(order) {
                _iter_rows = row_end - row_start;
                _iter_cols = col_end - col_start;
                if (_iter_rows < 0 || _iter_cols < 0)
                    throw std::invalid_argument("Stop rows or cols < start rows or cols");
                if (row_end > mat_ptr->rows() || col_end > mat_ptr->cols())
                    throw std::invalid_argument("Stop rows or cols out of matrix dims");
                _iter_size = _iter_rows * _iter_cols;
            }
            MatrixIterator& operator++() {
                ++_ptr;
                return *this;
            }
            MatrixIterator operator++(int) {
                MatrixIterator retval = *this;
                ++(*this);
                return retval;
            }
            MatrixIterator& operator--() {
                --_ptr;
                return *this;
            }
            MatrixIterator operator--(int) {
                MatrixIterator retval = *this;
                --(*this);
                return retval;
            }
            bool operator==(MatrixIterator iter) const {
                return _ptr == iter._ptr && _row_start == iter._row_start &&
                       _col_start == iter._col_start && _row_end == iter._row_end &&
                       _col_end == iter._col_end && _order == iter._order;
            }
            bool operator!=(MatrixIterator iter) const {
                return !(*this == iter);
            }
            T& operator*() {
                std::size_t row = 0, col = 0;
                switch (_order) {
                case ROWS: {
                    row = _ptr / _iter_cols;
                    col = _ptr % _iter_cols;
                    break;
                }
                case COLS: {
                    row = _ptr % _iter_rows;
                    col = _ptr / _iter_rows;
                    break;
                }
                }
                return _mat_ptr->at(row, col);
            }
            const T& operator*() const { return *(*this); }
            T* operator->() { return &(*(*this)); }
        };

        MatrixIterator begin() const { return MatrixIterator(const_cast<Matrix*>(this), ROWS, 0, 0, _rows, 0, _cols); }
        MatrixIterator end() const { return MatrixIterator(const_cast<Matrix*>(this), ROWS, size(), 0, _rows, 0, _cols); }

#pragma endregion ITERATOR

        template<typename U>
        friend std::ostream& operator<<(std::ostream& stream, const Matrix<U>& object);
    };
    
    template<typename T>
    std::ostream& operator<<(std::ostream& stream, const Matrix<T>& object) {
        stream << '[';
        for (std::size_t i = 0; i < object._rows; i++) {
            if (i > 0) stream << ' ';
            stream << '[';
            for (std::size_t j = 0; j < object._cols; j++) {
                stream << object.get(i, j);
                if (j < object._cols - 1) stream << ' ';
            }
            stream << ']';
            if (i < object._rows - 1) stream << std::endl;
        }
        stream << "] Matrix [" << object._rows << ',' << object._cols << ']';
        return stream;
    }
};

#endif