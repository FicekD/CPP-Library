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

constexpr double PI = 3.141592653589793238463;

namespace ndarray {
	
	template <typename T>
	class BaseArray {
	protected:
		std::unique_ptr<T> _data = nullptr;
		std::size_t _size = 0;
	public:
        BaseArray() {}
		BaseArray(const BaseArray<T>& arr) : _size(arr._size) {
			if (_size > 0) {
				_data = std::unique_ptr<T>(new T[arr._size]);
				std::memcpy(_data.get(), arr._data.get(), arr._size * sizeof(T));
			}
		}
		BaseArray(BaseArray<T>&& arr) noexcept : _size(arr._size) {
			_data = std::move(arr._data);
			arr.clear();
		}
		BaseArray(std::size_t size) : _size(size) {
			if (_size > 0)
				_data = std::unique_ptr<T>(new T[_size]{ T() });
		}

        virtual ~BaseArray() = default;

		std::size_t size() const { return _size; }

		void clear() {
			_data.release();
			_size = 0;
		}
		void fill(const T& scalar) {
			for (std::size_t i = 0; i < _size; i++)
				this->at(i) = scalar;
		}

        bool empty() const {
            return _data == nullptr || _size == 0;
        }

		T get(std::size_t x) const {
			if (x >= _size)
				throw std::invalid_argument("Index out of bounds");
			return _data.get()[x];
		}
		T& at(std::size_t x) {
			if (x >= _size)
				throw std::invalid_argument("Index out of bounds");
			return _data.get()[x];
		}

        template <typename T1, typename T2>
        static void _except_on_size_mismatch(const BaseArray<T1>& arr1, const BaseArray<T2>& arr2) {
            if (arr1.size() != arr2.size())
                throw std::invalid_argument("Size mismatch");
        }
        void map_inplace(const std::function<T(const T&)>& lambda) {
            for (int i = 0; i < _size; i++) {
                T& ref = this->at(i);
                ref = lambda(ref);
            }
        }
        void map_inplace(const std::function<T(const T&, const T&)>& lambda, const BaseArray<T>& arr) {
            _except_on_size_mismatch<T, T>(*this, arr);
            for (int i = 0; i < _size; i++) {
                T& ref = this->at(i);
                ref = lambda(ref, arr.get(i));
            }
        }
        template <typename R = T>
        void map_to(const std::function<R(const T&)>& lambda, BaseArray<R>& target) const {
            _except_on_size_mismatch<T, R>(*this, target);
            for (int i = 0; i < _size; i++) {
                target.at(i) = lambda(this->get(i));
            }
        }
        template <typename R = T>
        void map_to(const std::function<R(const T&, const T&)>& lambda, const BaseArray<T>& arr, BaseArray<R>& target) const {
            _except_on_size_mismatch<T, T>(*this, arr);
            _except_on_size_mismatch<T, R>(*this, target);
            for (int i = 0; i < _size; i++) {
                target.at(i) = lambda(this->get(i), arr.get(i));
            }
        }

        void add_inplace(const T& scalar) {
            map_inplace([scalar](const T& x) -> T { return x + scalar; });
        }
        void add_inplace(const BaseArray<T>& arr) {
            map_inplace([](const T& x1, const T& x2) -> T { return x1 + x2; }, arr);
        }
        void subtract_inplace(const T& scalar) {
            map_inplace([scalar](const T& x) -> T { return x - scalar; });
        }
        void subtract_inplace(const BaseArray<T>& arr) {
            map_inplace([](const T& x1, const T& x2) -> T { return x1 - x2; }, arr);
        }
        void multiply_inplace(const T& scalar) {
            map_inplace([scalar](const T& x) -> T { return x * scalar; });
        }
        void multiply_inplace(const BaseArray<T>& arr) {
            map_inplace([](const T& x1, const T& x2) -> T { return x1 * x2; }, arr);
        }
        void divide_inplace(const T& scalar) {
            map_inplace([scalar](const T& x) -> T { return x / scalar; });
        }
        void divide_inplace(const BaseArray<T>& arr) {
            map_inplace([](const T& x1, const T& x2) -> T { return x1 / x2; }, arr);
        }
        void square_inplace() {
            map_inplace([](const T& x) -> T { return x * x; });
        }
        void sqrt_inplace() {
            map_inplace([](const T& x) -> T { return static_cast<T>(std::sqrt(x)); });
        }
        void pow_inplace(double power) {
            map_inplace([power](const T& x) -> T { return static_cast<T>(std::pow(x, power)); });
        }
        void exp_inplace() {
            map_inplace([](const T& x) -> T { return std::exp(x); });
        }
        void exp2_inplace() {
            map_inplace([](const T& x) -> T { return std::exp2(x); });
        }
        void exp10_inplace() {
            map_inplace([](const T& x) -> T { return std::pow(T(10), x); });
        }
        void log_inplace() {
            map_inplace([](const T& x) -> T { return std::log(x); });
        }
        void log2_inplace() {
            map_inplace([](const T& x) -> T { return std::log2(x); });
        }
        void log10_inplace() {
            map_inplace([](const T& x) -> T { return std::log10(x); });
        }

        void sin_inplace() {
            map_inplace([](const T& x) -> T { return std::sin(x); });
        }
        void cos_inplace() {
            map_inplace([](const T& x) -> T { return std::cos(x); });
        }
        void tan_inplace() {
            map_inplace([](const T& x) -> T { return std::tan(x); });
        }
        void arcsin_inplace() {
            map_inplace([](const T& x) -> T { return std::asin(x); });
        }
        void arccos_inplace() {
            map_inplace([](const T& x) -> T { return std::acos(x); });
        }
        void arctan_inplace() {
            map_inplace([](const T& x) -> T { return std::atan(x); });
        }
        void deg2rad_inplace() {
            map_inplace([](const T& x) -> T { return x * T(PI / 180.0); });
        }
        void rad2deg_inplace() {
            map_inplace([](const T& x) -> T { return x * T(180.0 / PI); });
        }

        void abs_inplace() {
            map_inplace([](const T& x) -> T { return std::abs(x); });
        }
        void clamp_inplace(const T& min, const T& max) {
            map_inplace([min, max](const T& x) -> T { return std::clamp(x, min, max); });
        }
        void round_inplace() {
            map_inplace([](const T& x) -> T { return std::round(x); });
        }
        void floor_inplace() {
            map_inplace([](const T& x) -> T { return std::floor(x); });
        }
        void ceil_inplace() {
            map_inplace([](const T& x) -> T { return std::ceil(x); });
        }
        void trunc_inplace() {
            map_inplace([](const T& x) -> T { return std::trunc(x); });
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