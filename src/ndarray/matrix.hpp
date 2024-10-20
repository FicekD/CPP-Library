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

#include "base_array.hpp"
#include "ndarray.hpp"

namespace ndarray {

    enum MatrixDim {
        ROWS, COLS
    };

    template <typename T>
    class Matrix : public NDArray<Matrix, T> {
    private:
        std::size_t _rows = 0, _cols = 0;
    public:

#pragma region CONSTRUCTORS
        Matrix() {}
        Matrix(const Matrix<T>& matrix) noexcept : _rows(matrix._rows), _cols(matrix._cols), NDArray<Matrix, T>(matrix.size()) {
            if (size() > 0) {
                BaseArray<T>::_data = std::unique_ptr<T[]>(new T[matrix.size()]);
                std::memcpy(BaseArray<T>::_data.get(), matrix.BaseArray<T>::_data.get(), matrix.size() * sizeof(T));
            }
        }
        Matrix(Matrix<T>&& matrix) noexcept : _rows(matrix._rows), _cols(matrix._cols), NDArray<Matrix, T>() {
            BaseArray<T>::_size = matrix.size();
            BaseArray<T>::_data = std::move(matrix.BaseArray<T>::_data);
            matrix.clear();
        }
        Matrix(std::size_t rows, std::size_t cols) noexcept : _rows(rows), _cols(cols), NDArray<Matrix, T>(cols * rows) {
            if (size() > 0)
                BaseArray<T>::_data = std::unique_ptr<T[]>(new T[size()] { T() });
        }
        Matrix(const Shape& shape) noexcept : _rows(shape[0]), _cols(shape[1]), NDArray<Matrix, T>(shape[0] * shape[1]) {
            if (size() > 0)
                BaseArray<T>::_data = std::unique_ptr<T[]>(new T[size()]{ T() });
        }
        Matrix(std::size_t rows, std::size_t cols, T* data) noexcept : _rows(rows), _cols(cols), NDArray<Matrix, T>(cols * rows) {
            if (size() > 0) {
                BaseArray<T>::_data = std::unique_ptr<T[]>(new T[size()]);
                std::memcpy(BaseArray<T>::_data.get(), data, size() * sizeof(T));
            }
        }
        Matrix(const Shape& shape, T* data) noexcept : _rows(shape[0]), _cols(shape[1]), NDArray<Matrix, T>(shape[0] * shape[1]) {
            if (size() > 0) {
                BaseArray<T>::_data = std::unique_ptr<T[]>(new T[size()]);
                std::memcpy(BaseArray<T>::_data.get(), data, size() * sizeof(T));
            }
        }
        Matrix(std::size_t rows, std::size_t cols, const std::vector<T>& data) noexcept : _rows(rows), _cols(cols), NDArray<Matrix, T>(cols * rows) {
            if (size() > 0) {
                BaseArray<T>::_data = std::unique_ptr<T[]>(new T[size()]);
                std::memcpy(BaseArray<T>::_data.get(), data.data(), size() * sizeof(T));
            }
        }
        Matrix(const Shape& shape, const std::vector<T>& data) noexcept : _rows(shape[0]), _cols(shape[1]), NDArray<Matrix, T>(shape[0] * shape[1]) {
            if (size() > 0) {
                BaseArray<T>::_data = std::unique_ptr<T[]>(new T[size()]);
                std::memcpy(BaseArray<T>::_data.get(), data.data(), size() * sizeof(T));
            }
        }
        Matrix(const std::vector<Matrix<T>>& matrices, MatrixDim concat_dim) {
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
                BaseArray<T>::_data = std::unique_ptr<T[]>(new T[size()]);
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
                BaseArray<T>::_data = std::unique_ptr<T[]>(new T[size()]);
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
        using BaseArray<T>::empty;

        std::size_t rows() const { return _rows; }
        std::size_t cols() const { return _cols; }

        Shape shape() const {
            return Shape(_rows, _cols);
        }

        void clear() {
            _rows = 0;
            _cols = 0;
            BaseArray<T>::clear();
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

        template <typename U>
        Matrix<U> astype() const {
            Matrix<U> result(rows(), cols());
            for (int i = 0; i < size(); i++)
                result.at(i) = static_cast<U>(this->get(i));
            return result;
        }

        void reshape(std::size_t rows, std::size_t cols) {
            if (rows * cols != size())
                throw std::invalid_argument(std::format("Reshape missmatch [{}x{}] -> [{}x{}]", _rows, _cols, rows, cols));
            _rows = rows;
            _cols = cols;
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

        void operator=(const Matrix<T>& matrix) noexcept {
            if (matrix.size() != size()) {
                BaseArray<T>::_data.release();
                BaseArray<T>::_data = std::unique_ptr<T[]>(new T[matrix.size()]);
            }
            std::memcpy(BaseArray<T>::_data.get(), matrix.BaseArray<T>::_data.get(), matrix.size() * sizeof(T));
            BaseArray<T>::_size = matrix.size();
            _rows = matrix._rows;
            _cols = matrix._cols;
        }

#pragma endregion CORE
        
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
            Matrix<R> result(_rows, 1);
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
        Matrix<R> reduce(const std::function<R(const R&, const T&)>& lambda, std::size_t dim, const R& initializer) const {
            if (this->size() == 0) {
                throw std::logic_error("Cannot reduce zero-sized array");
            }
            if (dim == 0) {
                return reduce_rows<R>(lambda, initializer);
            }
            else if (dim == 1) {
                return reduce_cols<R>(lambda, initializer);
            }
            else {
                throw std::invalid_argument("Dim not valid for a Matrix type");
            }
        }

        Matrix<T> reduce_sum(MatrixDim dim) const {
            return reduce<T>(BaseArray<T>::reduce_sum_t.lambda, dim, BaseArray<T>::reduce_sum_t.initializer);
        }
        Matrix<T> reduce_prod(MatrixDim dim) const {
            return reduce<T>(BaseArray<T>::reduce_prod_t.lambda, dim, BaseArray<T>::reduce_prod_t.initializer);
        }
        Matrix<T> reduce_max(MatrixDim dim) const {
            return reduce<T>(BaseArray<T>::reduce_max_t.lambda, dim, BaseArray<T>::reduce_max_t.initializer);
        }
        Matrix<T> reduce_min(MatrixDim dim) const {
            return reduce<T>(BaseArray<T>::reduce_min_t.lambda, dim, BaseArray<T>::reduce_min_t.initializer);
        }
        Matrix<bool> reduce_any(MatrixDim dim) const {
            return reduce<bool>(BaseArray<T>::reduce_any_t.lambda, dim, BaseArray<T>::reduce_any_t.initializer);
        }
        Matrix<bool> reduce_all(MatrixDim dim) const {
            return reduce<bool>(BaseArray<T>::reduce_all_t.lambda, dim, BaseArray<T>::reduce_all_t.initializer);
        }

#pragma endregion REDUCE

#pragma region OTHER

        void swap_rows(size_t row1, size_t row2) {
            for (size_t col = 0; col < _cols; col++) {
                std::swap(this->at(row1, col), this->at(row2, col));
            }
        }
        void swap_cols(size_t col1, size_t col2) {
            for (size_t row = 0; row < _rows; row++) {
                std::swap(this->at(row, col1), this->at(row, col2));
            }
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

#pragma endregion OTHER

#pragma region ITERATOR
        class MatrixIterator {
        private:
            Matrix<T>* _mat_ptr;
            std::size_t _ptr;
            std::size_t _row_start, _col_start, _row_end, _col_end;
            std::size_t _iter_rows, _iter_cols, _iter_size;
            MatrixDim _order;
        public:
            using difference_type = std::ptrdiff_t;
            using element_type = T;
            using pointer = element_type*;
            using reference = element_type&;

            MatrixIterator(Matrix<T>* mat_ptr, MatrixDim order, std::size_t ptr,
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