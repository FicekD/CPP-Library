#ifndef _MATRIX_H
#define _MATRIX_H

#include <iostream>
#include <vector>
#include <memory>
#include <cstring>
#include <cmath>
#include <numeric>
#include <format>
#include <functional>

namespace ndarray {
    enum Dim {
        ROWS, COLS
    };

    template <typename T>
    class Matrix {
    private:
        std::unique_ptr<T> _data = nullptr;
        std::size_t _rows = 0, _cols = 0;
        std::size_t _size = 0;
    public:
        Matrix() {}
        Matrix(const Matrix<T>& matrix) : _rows(matrix._rows), _cols(matrix._cols), _size(matrix._size) {
            if (_size > 0) {
                _data = std::unique_ptr<T>(new T[matrix._size]);
                std::memcpy(_data.get(), matrix._data.get(), matrix._size * sizeof(T));
            }
        }
        Matrix(Matrix<T>&& matrix) noexcept : _rows(matrix._rows), _cols(matrix._cols), _size(matrix._size) {
            _data = std::move(matrix._data);
            matrix.clear();
        }
        Matrix(std::size_t rows, std::size_t cols) : _rows(rows), _cols(cols), _size(cols * rows) {
            if (_size > 0)
                _data = std::unique_ptr<T>(new T[_size] { T() });
        }
        Matrix(std::size_t rows, std::size_t cols, T* data) : _rows(rows), _cols(cols), _size(cols * rows) {
            if (_size > 0) {
                _data = std::unique_ptr<T>(new T[_size]);
                std::memcpy(_data.get(), data, _size * sizeof(T));
            }
        }
        Matrix(std::size_t rows, std::size_t cols, const std::vector<T>& data) : _rows(rows), _cols(cols), _size(cols * rows) {
            if (_size > 0) {
                _data = std::unique_ptr<T>(new T[_size]);
                std::memcpy(_data.get(), data.data(), _size * sizeof(T));
            }
        }

        static Matrix<T> eye(std::size_t dim_size, int k = 0) {
            size_t abs_k = std::abs(k);
            if (abs_k > dim_size)
                throw std::invalid_argument("k out of eye matrix dimensions");
            Matrix<T> ret = Matrix(dim_size, dim_size);
            if (k >= 0)
                for (size_t i = 0; i < dim_size - abs_k; i++) {
                    ret.at(i, i + abs_k) = T(1);
                }
            else
                for (size_t i = 0; i < dim_size - abs_k; i++) {
                    ret.at(i + abs_k, i) = T(1);
                }
            return ret;
        }

        std::size_t rows() const { return _rows; }
        std::size_t cols() const { return _cols; }
        std::size_t size() const { return _size; }

        void clear() {
            _data.release();
            _size = 0; _rows = 0; _cols = 0;
        }
        void fill(const T& scalar) {
            for (std::size_t i = 0; i < _size; i++)
                this->at(i) = scalar;
        }

        T get(std::size_t x) const {
            if (x >= _size)
                throw std::invalid_argument("Indices out of bounds");
            return _data.get()[x];
        }
        T get(std::size_t row, std::size_t col) const {
            if (row >= _rows || col >= _cols)
                throw std::invalid_argument("Indices out of bounds");
            return get(row * _cols + col);
        }
        T& at(std::size_t x) {
            if (x >= _size)
                throw std::invalid_argument("Indices out of bounds");
            return _data.get()[x];
        }
        T& at(std::size_t row, std::size_t col) {
            if (row >= _rows || col >= _cols)
                throw std::invalid_argument("Indices out of bounds");
            return at(row * _cols + col);
        }

        void reshape(std::size_t rows, std::size_t cols) {
            if (rows * cols != _size)
                throw std::invalid_argument(std::format("Reshape missmatch [{}x{}] -> [{}x{}]", _rows, _cols, rows, cols));
            _rows = rows;
            _cols = cols;
        }
        
        template<typename U>
        Matrix<U> astype() {
            Matrix<U> result(_rows, _cols);
            for (int i = 0; i < _size; i++)
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

#pragma region ARITHMETIC_OPS
        void operator=(const Matrix<T>& matrix) {
            if (matrix._size != _size)
                _data = std::unique_ptr<T>(new T[matrix._size]);
            std::memcpy(_data.get(), matrix._data.get(), matrix._size * sizeof(T));
            _size = matrix._size;
            _rows = matrix._rows;
            _cols = matrix._cols;
        }
        Matrix<T> operator+(const Matrix<T>& matrix) const {
            if (matrix._rows != _rows || matrix._cols != _cols)
                throw std::invalid_argument("Matricies shape missmatch");
            Matrix<T> result(_rows, _cols);
            for (std::size_t i = 0; i < _size; i++)
                result.at(i) = this->get(i) + matrix.get(i);
            return result;
        }
        Matrix<T> operator+(const T& scalar) const {
            Matrix<T> result(_rows, _cols);
            for (std::size_t i = 0; i < _size; i++)
                result.at(i) = this->get(i) + scalar;
            return result;
        }
        Matrix<T> operator+() const {
            Matrix<T> result(*this);
            return result;
        }
        Matrix<T> operator-(const Matrix<T>& matrix) const {
            if (matrix._rows != _rows || matrix._cols != _cols)
                throw std::invalid_argument("Matricies shape missmatch");
            Matrix<T> result(_rows, _cols);
            for (std::size_t i = 0; i < _size; i++)
                result.at(i) = this->get(i) - matrix.get(i);
            return result;
        }
        Matrix<T> operator-(const T& scalar) const {
            Matrix<T> result(_rows, _cols);
            for (std::size_t i = 0; i < _size; i++)
                result.at(i) = this->get(i) - scalar;
            return result;
        }
        Matrix<T> operator-() const {
            Matrix<T> result(_rows, _cols);
            for (std::size_t i = 0; i < _size; i++)
                result.at(i) = -(this->get(i));
            return result;
        }
        Matrix<T> operator*(const Matrix<T>& matrix) const {
            if (matrix._rows != _rows || matrix._cols != _cols)
                throw std::invalid_argument("Matricies shape missmatch");
            Matrix<T> result(_rows, _cols);
            for (std::size_t i = 0; i < _size; i++)
                result.at(i) = this->get(i) * matrix.get(i);
            return result;
        }
        Matrix<T> operator*(const T& scalar) const {
            Matrix<T> result(_rows, _cols);
            for (std::size_t i = 0; i < _size; i++)
                result.at(i) = this->get(i) * scalar;
            return result;
        }
        Matrix<T> operator/(const Matrix<T>& matrix) const {
            if (matrix._rows != _rows || matrix._cols != _cols)
                throw std::invalid_argument("Matricies shape missmatch");
            Matrix<T> result(_rows, _cols);
            for (std::size_t i = 0; i < _size; i++)
                result.at(i) = this->get(i) / matrix.get(i);
            return result;
        }
        Matrix<T> operator/(const T& scalar) const {
            Matrix<T> result(_rows, _cols);
            for (std::size_t i = 0; i < _size; i++)
                result.at(i) = this->get(i) / scalar;
            return result;
        }
        Matrix<T> square() const {
            Matrix<T> result(_rows, _cols);
            for (std::size_t i = 0; i < _size; i++)
                result.at(i) = this->get(i) * this->get(i);
            return result;
        }
        Matrix<T> sqrt() const {
            Matrix<T> result(_rows, _cols);
            for (std::size_t i = 0; i < _size; i++)
                result.at(i) = static_cast<T>(std::sqrt(this->get(i)));
            return result;
        }
        Matrix<T> pow(double power) const {
            Matrix<T> result(_rows, _cols);
            for (std::size_t i = 0; i < _size; i++)
                result.at(i) = static_cast<T>(std::pow(this->get(i), power));
            return result;
        }
#pragma endregion ARITHMETIC_OPS
        
#pragma region INPLACE_ARITHMETIC_OPS
        void add_inplace(const T& scalar) {
            for (int i = 0; i < _size; i++) {
                this->at(i) += scalar;
            }
        }
        void add_inplace(const Matrix<T>& matrix) {
            if (matrix._rows != _rows || matrix._cols != _cols)
                throw std::invalid_argument("Matricies shape missmatch");
            for (int i = 0; i < _size; i++) {
                this->at(i) += matrix.get(i);
            }
        }
        void subtract_inplace(const T& scalar) {
            for (int i = 0; i < _size; i++) {
                this->at(i) -= scalar;
            }
        }
        void subtract_inplace(const Matrix<T>& matrix) {
            if (matrix._rows != _rows || matrix._cols != _cols)
                throw std::invalid_argument("Matricies shape missmatch");
            for (int i = 0; i < _size; i++) {
                this->at(i) -= matrix.get(i);
            }
        }
        void multiply_inplace(const T& scalar) {
            for (int i = 0; i < _size; i++) {
                this->at(i) *= scalar;
            }
        }
        void multiply_inplace(const Matrix<T>& matrix) {
            if (matrix._rows != _rows || matrix._cols != _cols)
                throw std::invalid_argument("Matricies shape missmatch");
            for (int i = 0; i < _size; i++) {
                this->at(i) *= matrix.get(i);
            }
        }
        void divide_inplace(const T& scalar) {
            for (int i = 0; i < _size; i++) {
                this->at(i) /= scalar;
            }
        }
        void divide_inplace(const Matrix<T>& matrix) {
            if (matrix._rows != _rows || matrix._cols != _cols)
                throw std::invalid_argument("Matricies shape missmatch");
            for (int i = 0; i < _size; i++) {
                this->at(i) /= matrix.get(i);
            }
        }
        void square_inplace() {
            for (int i = 0; i < _size; i++) {
                T& at_ref = this->at(i);
                at_ref = at_ref * at_ref;
            }
        }
        void sqrt_inplace() {
            for (int i = 0; i < _size; i++) {
                T& at_ref = this->at(i);
                at_ref = static_cast<T>(std::sqrt(at_ref));
            }
        }
        void pow_inplace(double power) {
            for (int i = 0; i < _size; i++) {
                T& at_ref = this->at(i);
                at_ref = static_cast<T>(std::pow(at_ref, power));
            }
        }
#pragma endregion INPLACE_ARITHMETIC_OPS

#pragma region LOGICAL_OPS
        Matrix<bool> operator<(const Matrix<T>& matrix) const {
            if (matrix._rows != _rows || matrix._cols != _cols)
                throw std::invalid_argument("Matricies shape missmatch");
            Matrix<bool> result(_rows, _cols);
            for (std::size_t i = 0; i < _size; i++)
                result.at(i) = this->get(i) < matrix.get(i);
            return result;
        }
        Matrix<bool> operator<(const T& scalar) const {
            Matrix<bool> result(_rows, _cols);
            for (std::size_t i = 0; i < _size; i++)
                result.at(i) = this->get(i) < scalar;
            return result;
        }
        Matrix<bool> operator<=(const Matrix<T>& matrix) const {
            if (matrix._rows != _rows || matrix._cols != _cols)
                throw std::invalid_argument("Matricies shape missmatch");
            Matrix<bool> result(_rows, _cols);
            for (std::size_t i = 0; i < _size; i++)
                result.at(i) = this->get(i) <= matrix.get(i);
            return result;
        }
        Matrix<bool> operator<=(const T& scalar) const {
            Matrix<bool> result(_rows, _cols);
            for (std::size_t i = 0; i < _size; i++)
                result.at(i) = this->get(i) <= scalar;
            return result;
        }
        Matrix<bool> operator>(const Matrix<T>& matrix) const {
            if (matrix._rows != _rows || matrix._cols != _cols)
                throw std::invalid_argument("Matricies shape missmatch");
            Matrix<bool> result(_rows, _cols);
            for (std::size_t i = 0; i > _size; i++)
                result.at(i) = this->get(i) > matrix.get(i);
            return result;
        }
        Matrix<bool> operator>(const T& scalar) const {
            Matrix<bool> result(_rows, _cols);
            for (std::size_t i = 0; i > _size; i++)
                result.at(i) = this->get(i) > scalar;
            return result;
        }
        Matrix<bool> operator>=(const Matrix<T>& matrix) const {
            if (matrix._rows != _rows || matrix._cols != _cols)
                throw std::invalid_argument("Matricies shape missmatch");
            Matrix<bool> result(_rows, _cols);
            for (std::size_t i = 0; i < _size; i++)
                result.at(i) = this->get(i) >= matrix.get(i);
            return result;
        }
        Matrix<bool> operator>=(const T& scalar) const {
            Matrix<bool> result(_rows, _cols);
            for (std::size_t i = 0; i < _size; i++)
                result.at(i) = this->get(i) >= scalar;
            return result;
        }
        Matrix<bool> operator==(const Matrix<T>& matrix) const {
            if (matrix._rows != _rows || matrix._cols != _cols)
                throw std::invalid_argument("Matricies shape missmatch");
            Matrix<bool> result(_rows, _cols);
            for (std::size_t i = 0; i < _size; i++)
                result.at(i) = this->get(i) == matrix.get(i);
            return result;
        }
        Matrix<bool> operator==(const T& scalar) const {
            Matrix<bool> result(_rows, _cols);
            for (std::size_t i = 0; i < _size; i++)
                result.at(i) = this->get(i) == scalar;
            return result;
        }
        Matrix<bool> operator!=(const Matrix<T>& matrix) const {
            if (matrix._rows != _rows || matrix._cols != _cols)
                throw std::invalid_argument("Matricies shape missmatch");
            Matrix<bool> result(_rows, _cols);
            for (std::size_t i = 0; i < _size; i++)
                result.at(i) = this->get(i) != matrix.get(i);
            return result;
        }
        Matrix<bool> operator!=(const T& scalar) const {
            Matrix<bool> result(_rows, _cols);
            for (std::size_t i = 0; i < _size; i++)
                result.at(i) = this->get(i) != scalar;
            return result;
        }
        Matrix<bool> operator^(const Matrix<bool>& matrix) const {
            if (matrix.rows() != _rows || matrix.cols() != _cols)
                throw std::invalid_argument("Matricies shape missmatch");
            Matrix<bool> result(_rows, _cols);
            for (std::size_t i = 0; i < _size; i++)
                result.at(i) = this->get(i) ^ matrix.get(i);
            return result;
        }
        Matrix<bool> operator^(bool scalar) const {
            Matrix<bool> result(_rows, _cols);
            for (std::size_t i = 0; i < _size; i++)
                result.at(i) = this->get(i) ^ scalar;
            return result;
        }
        Matrix<bool> operator||(const Matrix<bool>& matrix) const {
            if (matrix.rows() != _rows || matrix.cols() != _cols)
                throw std::invalid_argument("Matricies shape missmatch");
            Matrix<bool> result(_rows, _cols);
            for (std::size_t i = 0; i < _size; i++)
                result.at(i) = this->get(i) || matrix.get(i);
            return result;
        }
        Matrix<bool> operator||(bool scalar) const {
            Matrix<bool> result(_rows, _cols);
            for (std::size_t i = 0; i < _size; i++)
                result.at(i) = this->get(i) || scalar;
            return result;
        }
        Matrix<bool> operator&&(const Matrix<bool>& matrix) const {
            if (matrix.rows() != _rows || matrix.cols() != _cols)
                throw std::invalid_argument("Matricies shape missmatch");
            Matrix<bool> result(_rows, _cols);
            for (std::size_t i = 0; i < _size; i++)
                result.at(i) = this->get(i) && matrix.get(i);
            return result;
        }
        Matrix<bool> operator&&(bool scalar) const {
            Matrix<bool> result(_rows, _cols);
            for (std::size_t i = 0; i < _size; i++)
                result.at(i) = this->get(i) && scalar;
            return result;
        }
        Matrix<bool> operator!() const {
            Matrix<bool> result(_rows, _cols);
            for (std::size_t i = 0; i < _size; i++)
                result.at(i) = !this->get(i);
            return result;
        }
#pragma endregion LOGICAL_OPS
        template<typename R>
        Matrix<R> reduce_rows(const std::function<R(const R&, const T&)>& lambda, const R& initializer) const {
            Matrix<R> result(1, _cols);
            result.fill(initializer);
            for (std::size_t row = 0; row < _rows; row++) {
                for (std::size_t col = 0; col < _cols; col++) {
                    R& at_col = result.at(col);
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
                    R& at_row = result.at(row);
                    at_row = lambda(at_row, this->get(row, col));
                }
            }
            return result;
        }
        template<typename R>
        Matrix<R> reduce_dim(const std::function<R(const R&, const T&)>& lambda, Dim dim, const R& initializer) const {
            if (dim == ROWS) {
                return reduce_rows<R>(lambda, initializer);
            }
            else {
                return reduce_cols<R>(lambda, initializer);
            }
        }
        T reduce_sum() const {
            return std::reduce(_data.get(), _data.get() + _size, T(0), [](const T& x0, const T& x1) -> T { return x0 + x1; });
        }
        Matrix<T> reduce_sum_dim(Dim dim) const {
            auto reduce_func = [](const T& x0, const T& x1) -> T { return x0 + x1; };
            return reduce_dim<T>(reduce_func, dim, T(0));
        }
        T reduce_prod() const {
            return std::reduce(_data.get(), _data.get() + _size, T(1), [](const T& x0, const T& x1) -> T { return x0 * x1; });
        }
        Matrix<T> reduce_prod_dim(Dim dim) const {
            auto reduce_func = [](const T& x0, const T& x1) -> T { return x0 * x1; };
            return reduce_dim<T>(reduce_func, dim, T(1));
        }
        T reduce_max() const {
            return std::reduce(_data.get(), _data.get() + _size, T(-INFINITY), [](const T& x0, const T& x1) -> T { return x0 > x1 ? x0 : x1; });
        }
        Matrix<T> reduce_max_dim(Dim dim) const {
            auto reduce_func = [](const T& x0, const T& x1) -> T { return x0 > x1 ? x0 : x1; };
            return reduce_dim<T>(reduce_func, dim, T(-INFINITY));
        }
        T reduce_min() const {
            return std::reduce(_data.get(), _data.get() + _size, T(INFINITY), [](const T& x0, const T& x1) -> T { return x0 < x1 ? x0 : x1; });
        }
        Matrix<T> reduce_min_dim(Dim dim) const {
            auto reduce_func = [](const T& x0, const T& x1) -> T { return x0 < x1 ? x0 : x1; };
            return reduce_dim<T>(reduce_func, dim, T(INFINITY));
        }
        bool reduce_any() const {
            return std::reduce(_data.get(), _data.get() + _size, false, [](bool x0, const T& x1) -> T { return x0 || x1 != 0; });
        }
        Matrix<bool> reduce_any_dim(Dim dim) const {
            auto reduce_func = [](const T& x0, const T& x1) -> T { return x0 || x1 != 0; };
            return reduce_dim<bool>(reduce_func, dim, false);
        }
        bool reduce_all() const {
            return std::reduce(_data.get(), _data.get() + _size, true, [](bool x0, const T& x1) -> T { return x0 && x1 != 0; });
        }
        Matrix<bool> reduce_all_dim(Dim dim) const {
            auto reduce_func = [](const T& x0, const T& x1) -> T { return x0 && x1 != 0; };
            return reduce_dim<bool>(reduce_func, dim, true);
        }

        Matrix<T> transpose() const;
        void transpose_inplace();
        Matrix<T> dot(const Matrix<T>& matrix) const;
        Matrix<T> inverse() const;

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
                : _row_start(row_start), _row_end(row_end), _col_start(col_start), _col_end(col_end), _order(order), _ptr(ptr), _mat_ptr(mat_ptr) {
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
        MatrixIterator end() const { return MatrixIterator(const_cast<Matrix*>(this), ROWS, _size, 0, _rows, 0, _cols); }

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