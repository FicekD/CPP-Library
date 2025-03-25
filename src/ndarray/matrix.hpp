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

#include "range.hpp"

namespace ndarray {

    template <typename T = std::size_t>
    class Point2 {
    public:
        T X = T(0);
        T Y = T(0);

        Point2() {}
        Point2(const Point2& point) : X(point.X), Y(point.Y) {}
        Point2(Point2&& point) : X(point.X), Y(point.Y) { point.X = 0; point.Y = 0; }
        Point2(T X, T Y) : X(X), Y(Y) {}
    };

    enum MatrixDim {
        ROWS, COLS
    };

    template <typename T>
    class Matrix : public NDArray<Matrix, T> {
    private:
        std::size_t _total_rows = 0, _total_cols = 0;
        std::size_t _rows = 0, _cols = 0;

        std::size_t _row_stride = 1, _col_stride = 1;
        std::size_t _row_offset = 0, _col_offset = 0;

        std::size_t ravel_indices_view(std::size_t row, std::size_t col) const {
            return (_row_offset + row * _row_stride) * _total_cols + (_col_offset + col * _col_stride);
        }

        template<typename U>
        friend void copy_data(Matrix<T>& dst, const Matrix<T>& src, std::size_t size, std::size_t dst_offset = 0, std::size_t src_offset = 0);

    public:
        Matrix() {}
        Matrix(const Matrix<T>& matrix) noexcept : _total_rows(matrix.rows()), _total_cols(matrix.cols()), _rows(matrix.rows()), _cols(matrix.cols()), NDArray<Matrix, T>(matrix.size()) {
            if (size() > 0) {
                BaseArray<T>::_data = std::shared_ptr<T[]>(new T[matrix.size()]);
                copy_data(*this, matrix, matrix.size());
            }
        }
        Matrix(Matrix<T>&& matrix) : _total_rows(matrix.rows()), _total_cols(matrix.cols()), _rows(matrix.rows()), _cols(matrix.cols()), NDArray<Matrix, T>() {
            if (is_view())
                throw std::logic_error("Cannot move a view");
            BaseArray<T>::_size = matrix.size();
            BaseArray<T>::_data = std::move(matrix.BaseArray<T>::_data);
            matrix.clear();
        }
        Matrix(std::size_t rows, std::size_t cols) noexcept : _total_rows(rows), _total_cols(cols), _rows(rows), _cols(cols), NDArray<Matrix, T>(cols * rows) {
            if (size() > 0)
                BaseArray<T>::_data = std::shared_ptr<T[]>(new T[size()] { T() });
        }
        Matrix(const Shape& shape) noexcept : _total_rows(shape[0]), _total_cols(shape[1]), _rows(shape[0]), _cols(shape[1]), NDArray<Matrix, T>(shape[0] * shape[1]) {
            if (size() > 0)
                BaseArray<T>::_data = std::shared_ptr<T[]>(new T[size()]{ T() });
        }
        Matrix(std::size_t rows, std::size_t cols, std::shared_ptr<T[]> data) noexcept : _total_rows(rows), _total_cols(cols), _rows(rows), _cols(cols), NDArray<Matrix, T>(cols * rows) {
            if (size() > 0)
                BaseArray<T>::_data = std::shared_ptr<T[]>(data);
        }
        Matrix(std::size_t rows, std::size_t cols, T* data) noexcept : _total_rows(rows), _total_cols(cols), _rows(rows), _cols(cols), NDArray<Matrix, T>(cols * rows) {
            if (size() > 0) {
                BaseArray<T>::_data = std::shared_ptr<T[]>(new T[size()]);
                std::memcpy(ptr(), data, size() * sizeof(T));
            }
        }
        Matrix(const Shape& shape, T* data) noexcept : _total_rows(shape[0]), _total_cols(shape[1]), _rows(shape[0]), _cols(shape[1]), NDArray<Matrix, T>(shape[0] * shape[1]) {
            if (size() > 0) {
                BaseArray<T>::_data = std::shared_ptr<T[]>(new T[size()]);
                std::memcpy(ptr(), data, size() * sizeof(T));
            }
        }
        Matrix(std::size_t rows, std::size_t cols, const std::vector<T>& data) noexcept : _total_rows(rows), _total_cols(cols), _rows(rows), _cols(cols), NDArray<Matrix, T>(cols * rows) {
            if (size() > 0) {
                BaseArray<T>::_data = std::shared_ptr<T[]>(new T[size()]);
                std::memcpy(ptr(), data.data(), size() * sizeof(T));
            }
        }
        Matrix(const Shape& shape, const std::vector<T>& data) noexcept : _total_rows(shape[0]), _total_cols(shape[1]), _rows(shape[0]), _cols(shape[1]), NDArray<Matrix, T>(shape[0] * shape[1]) {
            if (size() > 0) {
                BaseArray<T>::_data = std::shared_ptr<T[]>(new T[size()]);
                std::memcpy(ptr(), data.data(), size() * sizeof(T));
            }
        }
        Matrix(const std::vector<Matrix<T>>& matrices, MatrixDim concat_dim) {
            if (matrices.size() == 0)
                throw std::invalid_argument("Empty matrix vector to concat");
            
            if (concat_dim == ROWS) {
                _total_cols = matrices[0].cols();
                _total_rows = 0;
                for (const Matrix<T>& mat : matrices) {
                    if (mat.cols() != _total_cols)
                        throw std::invalid_argument("Matrix column missmatch");
                    _total_rows += mat.rows();
                }
                BaseArray<T>::_size = _total_rows * _total_cols;
                BaseArray<T>::_data = std::shared_ptr<T[]>(new T[size()]);
                std::size_t ptr_i = 0;
                for (const Matrix<T>& mat : matrices) {
                    copy_data(*this, mat, mat.size(), ptr_i);
                    ptr_i += mat.size();
                }
            }
            else {
                _total_cols = 0;
                _total_rows = matrices[0].rows();
                for (const Matrix<T>& mat : matrices) {
                    if (mat.rows() != _total_rows)
                        throw std::invalid_argument("Matrix row missmatch");
                    _total_cols += mat.cols();
                }
                BaseArray<T>::_size = _total_rows * _total_cols;
                BaseArray<T>::_data = std::shared_ptr<T[]>(new T[size()]);
                std::size_t ptr_i = 0;
                for (std::size_t row = 0; row < _total_rows; row++) {
                    for (const Matrix<T>& mat : matrices) {
                        copy_data(*this, mat, mat.cols(), ptr_i, row * mat.cols());
                        ptr_i += mat._total_cols;
                    }
                }
            }
            _rows = _total_rows;
            _cols = _total_cols;
        }

        Matrix<T> copy() const {
            return Matrix<T>(*this);
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

        using BaseArray<T>::ptr;
        using BaseArray<T>::is_view;
        using BaseArray<T>::fill;
        using BaseArray<T>::empty;

        std::size_t size() const override {
            return rows() * cols();
        }

        std::size_t ravel_indices(std::size_t row, std::size_t col) const {
            return row * cols() + col;
        }
        std::size_t ravel_indices(const Point2<>& point) const {
            return ravel_indices(point.Y, point.X);
        }
        void unravel_index(std::size_t i, std::size_t& row, std::size_t& col) const {
            row = i / cols();
            col = i % cols();
        }
        Point2<> unravel_index(std::size_t i) const {
            return Point2<>(i % cols(), i / cols());
        }

        std::size_t rows() const { return _rows; }
        std::size_t cols() const { return _cols; }

        Shape shape() const {
            return Shape(rows(), cols());
        }

        void clear() {
            BaseArray<T>::clear();
            _total_rows = 0;
            _total_cols = 0;
            _row_stride = 1;
            _col_stride = 1;
            _row_offset = 0;
            _col_offset = 0;
            _rows = 0;
            _cols = 0;
        }

        T get(std::size_t i) const override {
            std::size_t row, col;
            unravel_index(i, row, col);
            return get(row, col);
        }
        T& at(std::size_t i) override {
            std::size_t row, col;
            unravel_index(i, row, col);
            return at(row, col);
        }

        T get(std::size_t row, std::size_t col) const {
            if (row >= rows() || col >= cols())
                throw std::invalid_argument("Indices out of bounds");
            return BaseArray<T>::get(ravel_indices_view(row, col));
        }
        T& at(std::size_t row, std::size_t col) {
            if (row >= rows() || col >= cols())
                throw std::invalid_argument("Indices out of bounds");
            return BaseArray<T>::at(ravel_indices_view(row, col));
        }

        T get(const Point2<>& point) const {
            return get(point.Y, point.X);
        }
        T& at(const Point2<>& point) {
            return at(point.Y, point.X);
        }

        template <typename U>
        Matrix<U> astype() const {
            Matrix<U> result(rows(), cols());
            for (int i = 0; i < size(); i++)
                result.at(i) = static_cast<U>(this->get(i));
            return result;
        }

        Matrix<T> reshape(std::size_t rows, std::size_t cols) {
            if (rows * cols != size())
                throw std::invalid_argument(std::format("Reshape missmatch [{}x{}] -> [{}x{}]", this->rows(), this->cols(), rows, cols));
            if (is_view()) {
                Matrix<T> ret(rows, cols);
                for (std::size_t i = 0; i < size(); i++)
                    ret.at(i) = get(i);
                return ret;
            }
            else {
                _total_rows = rows;
                _total_cols = cols;

                return *this;
            }
        }

        void operator=(const Matrix<T>& matrix) {
            if (empty()) {
                _total_rows = matrix._total_rows;
                _total_cols = matrix._total_cols;

                _row_stride = matrix._row_stride;
                _col_stride = matrix._col_stride;
                _row_offset = matrix._row_offset;
                _col_offset = matrix._col_offset;
                _rows = matrix._rows;
                _cols = matrix._cols;

                BaseArray<T>::_size = _total_rows * _total_cols;
                BaseArray<T>::_data = std::shared_ptr<T[]>(matrix.BaseArray<T>::_data);
            }
            else {
                if (matrix.rows() != rows() || matrix.cols() != cols())
                    throw std::invalid_argument("Matrix shape missmatch");
                copy_data(*this, matrix, matrix.size());
            }
        }

        Matrix<T> view(std::size_t row_start, std::size_t row_stride, std::size_t row_end, std::size_t col_start, std::size_t col_stride, std::size_t col_end) const {
            Matrix<T> view_mat(_total_rows, _total_cols, view_mat._data);

            view_mat._row_offset = _row_offset + _row_stride * row_start;
            view_mat._col_offset = _col_offset + _col_stride * row_start;

            view_mat._row_stride = _row_stride * row_stride;
            view_mat._col_stride = _col_stride * col_stride;

            view_mat._rows = (row_end - row_start - 1) / row_stride;
            view_mat._cols = (col_end - col_start - 1) / col_stride;

            return view_mat;
        }
        Matrix<T> view(std::size_t row_start, std::size_t row_end, std::size_t col_start, std::size_t col_end) const {
            return view(row_start, 1, row_end, col_start, 1, col_end);
        }
        Matrix<T> view(const range::Range& row_range, const range::Range& col_range) const {
            return view(row_range.start, row_range.stride, row_range.stop, col_range.start, col_range.stride, col_range.stop);
        }

        using BaseArray<T>::reduce_sum;
        using BaseArray<T>::reduce_prod;
        using BaseArray<T>::reduce_max;
        using BaseArray<T>::reduce_min;
        using BaseArray<T>::reduce_any;
        using BaseArray<T>::reduce_all;
        
        template<typename R>
        Matrix<R> reduce_rows(const std::function<R(const R&, const T&)>& lambda, const R& initializer) const {
            Matrix<R> result(1, cols());
            result.fill(initializer);
            for (std::size_t row = 0; row < rows(); row++) {
                for (std::size_t col = 0; col < cols(); col++) {
                    R& at_col = result.BaseArray<R>::at(col);
                    at_col = lambda(at_col, this->get(row, col));
                }
            }
            return result;
        }
        template<typename R>
        Matrix<R> reduce_cols(const std::function<R(const R&, const T&)>& lambda, const R& initializer) const {
            Matrix<R> result(rows(), 1);
            result.fill(initializer);
            for (std::size_t row = 0; row < rows(); row++) {
                for (std::size_t col = 0; col < cols(); col++) {
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

        void swap_rows(size_t row1, size_t row2) {
            for (size_t col = 0; col < cols(); col++) {
                std::swap(this->at(row1, col), this->at(row2, col));
            }
        }
        void swap_cols(size_t col1, size_t col2) {
            for (size_t row = 0; row < rows(); row++) {
                std::swap(this->at(row, col1), this->at(row, col2));
            }
        }
        Matrix<T> flip_ud() const {
            Matrix<T> result(rows(), cols());
            for (std::size_t row = 0; row < std::size_t(std::ceil(float(rows()) / 2)); row++) {
                for (std::size_t col = 0; col < cols(); col++) {
                    result.at(rows() - 1 - row, col) = this->get(row, col);
                    result.at(row, col) = this->get(rows() - 1 - row, col);
                }
            }
            return result;
        }
        Matrix<T> flip_lr() const {
            Matrix<T> result(rows(), cols());
            for (std::size_t col = 0; col < std::size_t(std::ceil(float(cols()) / 2)); col++) {
                for (std::size_t row = 0; row < rows(); row++) {
                    result.at(row, _total_cols - 1 - col) = this->get(row, col);
                    result.at(row, col) = this->get(row, _total_cols - 1 - col);
                }
            }
            return result;
        }
        
        Matrix<T> pad(std::size_t pre_rows, std::size_t post_rows, std::size_t pre_cols, std::size_t post_cols, const T& scalar) const {
            Matrix<T> result(rows() + pre_rows + post_rows, cols() + pre_cols + post_cols);
            for (std::size_t row = 0; row < pre_rows; row++)
                for (std::size_t col = 0; col < result.cols(); col++)
                    result.at(row, col) = scalar; 
            for (std::size_t row = pre_rows + rows(); row < result.rows(); row++)
                for (std::size_t col = 0; col < result.cols(); col++)
                    result.at(row, col) = scalar;
            for (std::size_t col = 0; col < pre_cols; col++)
                for (std::size_t row = pre_rows; row < result.rows() - post_rows; row++)
                    result.at(row, col) = scalar;
            for (std::size_t col = pre_cols + cols(); col < result.cols(); col++)
                for (std::size_t row = pre_rows; row < result.rows() - post_rows; row++)
                    result.at(row, col) = scalar;
            
            for (std::size_t row = 0; row < rows(); row++)
                for (std::size_t col = 0; col < cols(); col++)
                    result.at(row + pre_rows, col + pre_cols) = this->get(row, col);
            return result;
        }

        class MatrixIterator {
        private:
            Matrix<T>* _mat_ptr;
            std::size_t i;
            MatrixDim _order;
        public:
            using difference_type = std::ptrdiff_t;
            using element_type = T;
            using pointer = element_type*;
            using reference = element_type&;

            MatrixIterator(Matrix<T>* mat_ptr, std::size_t i, MatrixDim order = ROWS) : _mat_ptr(mat_ptr), i(i), _order(order) {}

            MatrixIterator& operator++() {
                ++i;
                return *this;
            }
            MatrixIterator operator++(int) {
                MatrixIterator retval = *this;
                ++(*this);
                return retval;
            }
            MatrixIterator& operator--() {
                --i;
                return *this;
            }
            MatrixIterator operator--(int) {
                MatrixIterator retval = *this;
                --(*this);
                return retval;
            }
            bool operator==(MatrixIterator iter) const {
                return i == iter.i && _order == iter._order;
            }
            bool operator!=(MatrixIterator iter) const {
                return !(*this == iter);
            }
            reference operator*() {
                std::size_t row = 0, col = 0;
                switch (_order) {
                case ROWS: {
                    row = i / _mat_ptr->cols();
                    col = i % _mat_ptr->cols();
                    break;
                }
                case COLS: {
                    row = i % _mat_ptr->rows();
                    col = i / _mat_ptr->rows();
                    break;
                }
                }
                return _mat_ptr->at(row, col);
            }
            const reference operator*() const { return *(*this); }
            pointer operator->() { return &(*(*this)); }
        };

        MatrixIterator begin() const { return MatrixIterator(const_cast<Matrix*>(this), 0, ROWS); }
        MatrixIterator end() const { return MatrixIterator(const_cast<Matrix*>(this), size(), ROWS); }

        template<typename U>
        friend std::ostream& operator<<(std::ostream& stream, const Matrix<U>& object);
    };
    
    template<typename T>
    std::ostream& operator<<(std::ostream& stream, const Matrix<T>& object) {
        stream << '[';
        for (std::size_t i = 0; i < object.rows(); i++) {
            if (i > 0) stream << ' ';
            stream << '[';
            for (std::size_t j = 0; j < object.cols(); j++) {
                stream << object.get(i, j);
                if (j < object.cols() - 1) stream << ' ';
            }
            stream << ']';
            if (i < object.rows() - 1) stream << std::endl;
        }
        stream << "] Matrix [" << object.rows() << ',' << object.cols() << ']';
        return stream;
    }

    template<typename T>
    void copy_data(Matrix<T>& dst, const Matrix<T>& src, std::size_t size, std::size_t dst_offset = 0, std::size_t src_offset = 0) {
        if (src.is_view() || dst.is_view()) {
            for (std::size_t i = 0; i < size; i++) {
                dst.at(i + dst_offset) = src.get(i + src_offset);
            }
        }
        else {
            std::memcpy(dst.ptr() + dst_offset, src.ptr() + src_offset, size * sizeof(T));
        }
    }
};

#endif