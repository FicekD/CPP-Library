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


namespace ndarray {
    constexpr double PI = 3.141592653589793238463;

    template <typename T, typename R>
    struct Reduce {
        std::function<R(const R&, const T&)> lambda;
        R initializer;
    };

	template <typename T>
	class BaseArray {
	protected:
		std::unique_ptr<T[]> _data = nullptr;
		std::size_t _size = 0;

        const Reduce<T, T> reduce_sum_t{ [](const T& x0, const T& x1) -> T { return x0 + x1; }, T(0) };
        const Reduce<T, T> reduce_prod_t{ [](const T& x0, const T& x1) -> T { return x0 * x1; }, T(1) };
        const Reduce<T, T> reduce_max_t{ [](const T& x0, const T& x1) -> T { return x0 > x1 ? x0 : x1; }, T(-INFINITY) };
        const Reduce<T, T> reduce_min_t{ [](const T& x0, const T& x1) -> T { return x0 < x1 ? x0 : x1; }, T(INFINITY) };
        const Reduce<T, bool> reduce_any_t{ [] (bool x0, const T& x1) -> bool { return x0 || x1; }, false };
        const Reduce<T, bool> reduce_all_t{ [] (bool x0, const T& x1) -> bool { return x0 && x1; }, true };

	public:
        BaseArray() {}
		BaseArray(const BaseArray<T>& arr) noexcept : _size(arr._size) {
			if (_size > 0) {
				_data = std::unique_ptr<T[]>(new T[arr._size]);
				std::memcpy(_data.get(), arr._data.get(), arr._size * sizeof(T));
			}
		}
		BaseArray(BaseArray<T>&& arr) noexcept : _size(arr._size) {
			_data = std::move(arr._data);
			arr.clear();
		}
		BaseArray(std::size_t size) noexcept : _size(size) {
			if (_size > 0)
				_data = std::unique_ptr<T[]>(new T[_size]{ T() });
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

        static void _except_on_size_mismatch(std::size_t size1, std::size_t size2) {
            if (size1 != size2)
                throw std::invalid_argument("Size mismatch");
        }
        void map_inplace(const std::function<T(const T&)>& lambda) {
            for (int i = 0; i < _size; i++) {
                T& ref = this->at(i);
                ref = lambda(ref);
            }
        }
        void map_inplace(const std::function<T(const T&, const T&)>& lambda, const BaseArray<T>& arr) {
            BaseArray<T>::_except_on_size_mismatch(this->size(), arr.size());
            for (int i = 0; i < _size; i++) {
                T& ref = this->at(i);
                ref = lambda(ref, arr.get(i));
            }
        }
        template <typename R = T>
        void map_to(const std::function<R(const T&)>& lambda, BaseArray<R>& target) const {
            BaseArray<T>::_except_on_size_mismatch(this->size(), target.size());
            for (int i = 0; i < _size; i++) {
                target.at(i) = lambda(this->get(i));
            }
        }
        template <typename R = T>
        void map_to(const std::function<R(const T&, const T&)>& lambda, const BaseArray<T>& arr, BaseArray<R>& target) const {
            BaseArray<T>::_except_on_size_mismatch(this->size(), arr.size());
            BaseArray<T>::_except_on_size_mismatch(this->size(), target.size());
            for (int i = 0; i < _size; i++) {
                target.at(i) = lambda(this->get(i), arr.get(i));
            }
        }

        void positive_inplace() {
            map_inplace([](const T& x) -> T { return +x; });
        }
        void negative_inplace() {
            map_inplace([](const T& x) -> T { return -x; });
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
        void pow_inplace(T power) {
            map_inplace([power](const T& x) -> T { return static_cast<T>(std::pow(x, power)); });
        }
        void pow_inplace(const BaseArray<T>& powers) {
            map_inplace([](const T& x1, const T& x2) -> T { return static_cast<T>(std::pow(x1, x2)); }, powers);
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
        void sign_inplace() {
            map_inplace([](const T& x) -> T { return (x > T(0)) - (x < T(0)); });
        }

        void less_inplace(const BaseArray<T>& arr) {
            map_inplace([](const T& x1, const T& x2) -> T { return x1 < x2; }, arr);
        }
        void less_inplace(const T& scalar) {
            map_inplace([scalar](const T& x) -> T { return x < scalar; });
        }
        void less_equal_inplace(const BaseArray<T>& arr) {
            map_inplace([](const T& x1, const T& x2) -> T { return x1 <= x2; }, arr);
        }
        void less_equal_inplace(const T& scalar) {
            map_inplace([scalar](const T& x) -> T { return x <= scalar; });
        }
        void greater_inplace(const BaseArray<T>& arr) {
            map_inplace([](const T& x1, const T& x2) -> T { return x1 > x2; }, arr);
        }
        void greater_inplace(const T& scalar) {
            map_inplace([scalar](const T& x) -> T { return x > scalar; });
        }
        void greater_equal_inplace(const BaseArray<T>& arr) {
            map_inplace([](const T& x1, const T& x2) -> T { return x1 >= x2; }, arr);
        }
        void greater_equal_inplace(const T& scalar) {
            map_inplace([scalar](const T& x) -> T { return x >= scalar; });
        }
        void equal_inplace(const BaseArray<T>& arr) {
            map_inplace([](const T& x1, const T& x2) -> T { return x1 == x2; }, arr);
        }
        void equal_inplace(const T& scalar) {
            map_inplace([scalar](const T& x) -> T { return x == scalar; });
        }
        void not_equal_inplace(const BaseArray<T>& arr) {
            map_inplace([](const T& x1, const T& x2) -> T { return x1 != x2; }, arr);
        }
        void not_equal_inplace(const T& scalar) {
            map_inplace([scalar](const T& x) -> T { return x != scalar; });
        }
        void logical_or_inplace(const BaseArray<T>& arr) {
            map_inplace([](const T& x1, const T& x2) -> T { return x1 || x2; }, arr);
        }
        void logical_or_inplace(const T& scalar) {
            map_inplace([scalar](const T& x) -> T { return x || scalar; });
        }
        void logical_and_inplace(const BaseArray<T>& arr) {
            map_inplace([](const T& x1, const T& x2) -> T { return x1 && x2; }, arr);
        }
        void logical_and_inplace(const T& scalar) {
            map_inplace([scalar](const T& x) -> T { return x && scalar; });
        }
        void logical_not_inplace() {
            map_inplace([](const T& x) -> T { return !x; });
        }

        void bitwise_or_inplace(const BaseArray<T>& arr) {
            map_inplace([](const T& x1, const T& x2) -> T { return x1 | x2; }, arr);
        }
        void bitwise_or_inplace(const T& scalar) {
            map_inplace([scalar](const T& x) -> T { return x | scalar; });
        }
        void bitwise_and_inplace(const BaseArray<T>& arr) {
            map_inplace([](const T& x1, const T& x2) -> T { return x1 & x2; }, arr);
        }
        void bitwise_and_inplace(const T& scalar) {
            map_inplace([scalar](const T& x) -> T { return x & scalar; });
        }
        void bitwise_xor_inplace(const BaseArray<T>& arr) {
            map_inplace([](const T& x1, const T& x2) -> T { return x1 ^ x2; }, arr);
        }
        void bitwise_xor_inplace(const T& scalar) {
            map_inplace([scalar](const T& x) -> T { return x ^ scalar; });
        }
        void bitwise_not_inplace() {
            map_inplace([](const T& x) -> T { return ~x; });
        }

        T reduce_sum() const {
            return std::reduce(_data.get(), _data.get() + _size, reduce_sum_t.initializer, reduce_sum_t.lambda);
        }
        T reduce_prod() const {
            return std::reduce(_data.get(), _data.get() + _size, reduce_prod_t.initializer, reduce_prod_t.lambda);
        }
        T reduce_max() const {
            return std::reduce(_data.get(), _data.get() + _size, reduce_max_t.initializer, reduce_max_t.lambda);
        }
        T reduce_min() const {
            return std::reduce(_data.get(), _data.get() + _size, reduce_min_t.initializer, reduce_min_t.lambda);
        }
        bool reduce_any() const {
            return std::reduce(_data.get(), _data.get() + _size, reduce_any_t.initializer, reduce_any_t.lambda);
        }
        bool reduce_all() const {
            return std::reduce(_data.get(), _data.get() + _size, reduce_all_t.initializer, reduce_all_t.lambda);
        }
	};
}

#endif