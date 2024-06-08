#ifndef _ARRAY_H
#define _ARRAY_H

#include <iostream>
#include <vector>
#include <memory>
#include <cstring>
#include <cmath>
#include <numeric>
#include <functional>
#include <format>
#include <algorithm>

constexpr double M_PI = 3.141592653589793238463;

namespace ndarray {
	
	template <typename T>
	class Array {
	private:
		std::unique_ptr<T> _data = nullptr;
		std::size_t _size = 0;
	public:
		Array() {}
		Array(const Arrat<T>& arr) : _size(arr.size) {
			if (_size > 0) {
				_data = std::unique_ptr<T>(new T[arr._size]);
				std::memcpy(_data.get(), arr._data.get(), arr._size * sizeof(T));
			}
		}
		Array(Array<T>&& arr) noexcept : _size(arr._size) {
			_data = std::move(arr._data);
			arr.clear();
		}
		Array(std::size_t size) : _size(size) {
			if (_size > 0)
				_data = std::unique_ptr<T>(new T[_size]{ T() });
		}

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
		T& at(std::size_t x) {
			if (x >= _size)
				throw std::invalid_argument("Indices out of bounds");
			return _data.get()[x];
		}

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

        void exp_inplace() {
            for (std::size_t i = 0; i < _size; i++) {
                T& ref = this->at(i);
                ref = std::exp(ref);
            }
        }
        void exp2_inplace() {
            for (std::size_t i = 0; i < _size; i++) {
                T& ref = this->at(i);
                ref = std::exp2(ref);
            }
        }
        void exp10_inplace() {
            T ten = T(10);
            for (std::size_t i = 0; i < _size; i++) {
                T& ref = this->at(i);
                ref = std::pow(ten, ref);
            }
        }
        void log_inplace() {
            for (std::size_t i = 0; i < _size; i++) {
                T& ref = this->at(i);
                ref = std::log(ref);
            }
        }
        void log2_inplace() {
            for (std::size_t i = 0; i < _size; i++) {
                T& ref = this->at(i);
                ref = std::log2(ref);
            }
        }
        void log10_inplace() {
            for (std::size_t i = 0; i < _size; i++) {
                T& ref = this->at(i);
                ref = std::log10(ref);
            }
        }

        void sin_inplace() {
            for (std::size_t i = 0; i < _size; i++) {
                T& ref = this->at(i);
                ref = std::sin(ref);
            }
        }
        void cos_inplace() {
            for (std::size_t i = 0; i < _size; i++) {
                T& ref = this->at(i);
                ref = std::cos(ref);
            }
        }
        void tan_inplace() {
            for (std::size_t i = 0; i < _size; i++) {
                T& ref = this->at(i);
                ref = std::tan(ref);
            }
        }
        void arcsin_inplace() {
            for (std::size_t i = 0; i < _size; i++) {
                T& ref = this->at(i);
                ref = std::asin(ref);
            }
        }
        void arccos_inplace() {
            for (std::size_t i = 0; i < _size; i++) {
                T& ref = this->at(i);
                ref = std::acos(ref);
            }
        }
        void arctan_inplace() {
            for (std::size_t i = 0; i < _size; i++) {
                T& ref = this->at(i);
                ref = std::atan(ref);
            }
        }

        void deg2rad_inplace() {
            T coeff = T(M_PI / 180.0);
            for (std::size_t i = 0; i < _size; i++) {
                T& ref = this->at(i);
                ref = ref * coeff;
            }
        }
        void rad2deg_inplace() {
            T coeff = T(180.0 / M_PI);
            for (std::size_t i = 0; i < _size; i++) {
                T& ref = this->at(i);
                ref = ref * coeff;
            }
        }

        void abs_inplace() {
            for (std::size_t i = 0; i < _size; i++) {
                T& ref = this->at(i);
                ref = std::abs(ref);
            }
        }
        void clamp_inplace(const T& min, const T& max) {
            for (std::size_t i = 0; i < _size; i++) {
                T& ref = this->at(i);
                ref = std::clamp(ref, min, max);
            }
        }

        void round_inplace() {
            for (std::size_t i = 0; i < _size; i++) {
                T& ref = this->at(i);
                ref = std::round(ref);
            }
        }
        void floor_inplace() {
            for (std::size_t i = 0; i < _size; i++) {
                T& ref = this->at(i);
                ref = std::floor(ref);
            }
        }
        void ceil_inplace() {
            for (std::size_t i = 0; i < _size; i++) {
                T& ref = this->at(i);
                ref = std::ceil(ref);
            }
        }
        void trunc_inplace() {
            for (std::size_t i = 0; i < _size; i++) {
                T& ref = this->at(i);
                ref = std::trunc(ref);
            }
        }

        T reduce_sum() const {
            return std::reduce(_data.get(), _data.get() + _size, T(0), [](const T& x0, const T& x1) -> T { return x0 + x1; });
        }
        T reduce_prod() const {
            return std::reduce(_data.get(), _data.get() + _size, T(1), [](const T& x0, const T& x1) -> T { return x0 * x1; });
        }
        T reduce_max() const {
            return std::reduce(_data.get(), _data.get() + _size, T(-INFINITY), [](const T& x0, const T& x1) -> T { return x0 > x1 ? x0 : x1; });
        }
        T reduce_min() const {
            return std::reduce(_data.get(), _data.get() + _size, T(INFINITY), [](const T& x0, const T& x1) -> T { return x0 < x1 ? x0 : x1; });
        }
        bool reduce_any() const {
            return std::reduce(_data.get(), _data.get() + _size, false, [](bool x0, const T& x1) -> T { return x0 || x1 != 0; });
        }
        bool reduce_all() const {
            return std::reduce(_data.get(), _data.get() + _size, true, [](bool x0, const T& x1) -> T { return x0 && x1 != 0; });
        }
	};
}

#endif