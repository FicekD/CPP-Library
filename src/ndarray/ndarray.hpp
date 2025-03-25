#pragma once

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


namespace ndarray {
    class Shape {
    public:
        static const std::size_t MAX_DIMS = 8;

    private:
        std::size_t d[Shape::MAX_DIMS];

    public:
        Shape(const Shape& shape) {
            std::memcpy(this->d, shape.d, Shape::MAX_DIMS * sizeof(std::size_t));
        }

        Shape(std::size_t d0 = 1,
              std::size_t d1 = 1,
              std::size_t d2 = 1,
              std::size_t d3 = 1,
              std::size_t d4 = 1,
              std::size_t d5 = 1,
              std::size_t d6 = 1,
              std::size_t d7 = 1
        ) {
            d[0] = d0;
            d[1] = d1;
            d[2] = d2;
            d[3] = d3;
            d[4] = d4;
            d[5] = d5;
            d[6] = d6;
            d[7] = d7;
        }

        Shape(std::size_t* d) {
            for (std::size_t i = 0; i < Shape::MAX_DIMS; i++) {
                this->d[i] = d[i];
            }
        }

        std::size_t size() const {
            std::size_t total_size = 1;
            for (std::size_t i = 0; i < Shape::MAX_DIMS; i++) {
                total_size *= d[i];
            }
            return total_size;
        }

        bool operator==(const Shape& shape) const {
            for (std::size_t i = 0; i < Shape::MAX_DIMS; i++) {
                if (this->d[i] != shape.d[i]) return false;
            }
            return true;
        }

        bool operator!=(const Shape& shape) const {
            return !(*this == shape);
        }

        std::size_t operator[](std::size_t dim) const {
            if (dim >= Shape::MAX_DIMS) {
                throw std::out_of_range("Requested dim larger than max dims");
            }
            return d[dim];
        }
    };

	template <template <typename> typename container_t, typename T>
	class NDArray : public BaseArray<T> {
    public:
        NDArray() noexcept : BaseArray<T>() {}
        NDArray(std::size_t size) noexcept : BaseArray<T>(size) {}

        using BaseArray<T>::size;

        static void _except_on_shape_mismatch(const Shape& shape1, const Shape& shape2) {
            if (shape1 != shape2)
                throw std::invalid_argument("Shape mismatch");
        }

        virtual Shape shape() const = 0;

        template <typename R = T>
        container_t<R> map_to_new(const std::function<R(const T&, const T&)>& lambda, const container_t<T>& obj) const {
            NDArray::_except_on_shape_mismatch(this->shape(), obj.shape());
            container_t<R> result(this->shape());
            BaseArray<T>::template map_to<R>(lambda, dynamic_cast<const BaseArray<T>&>(obj), dynamic_cast<BaseArray<R>&>(result));
            return result;
        }

        template <typename R = T>
        container_t<R> map_to_new(const std::function<R(const T&)>& lambda) const {
            container_t<R> result(this->shape());
            BaseArray<T>::template map_to<R>(lambda, dynamic_cast<BaseArray<R>&>(result));
            return result;
        }

        container_t<T> operator+(const container_t<T>& obj) const {
            return add(obj);
        }
        container_t<T> operator+(const T& scalar) const {
            return add(scalar);
        }
        container_t<T> operator+() const {
            return positive();
        }
        container_t<T> operator-(const container_t<T>& obj) const {
            return subtract(obj);
        }
        container_t<T> operator-(const T& scalar) const {
            return subtract(scalar);
        }
        container_t<T> operator-() const {
            return negative();
        }
        container_t<T> operator*(const container_t<T>& obj) const {
            return multiply(obj);
        }
        container_t<T> operator*(const T& scalar) const {
            return multiply(scalar);
        }
        container_t<T> operator/(const container_t<T>& obj) const {
            return divide(obj);
        }
        container_t<T> operator/(const T& scalar) const {
            return divide(scalar);
        }
        container_t<T> positive() const {
            return map_to_new<T>([](const T& x) -> T { return +x; });
        }
        container_t<T> negative() const {
            return map_to_new<T>([](const T& x) -> T { return -x; });
        }
        container_t<T> add(const container_t<T>& obj) const {
            return map_to_new<T>([](const T& x1, const T& x2) -> T { return x1 + x2; }, obj);
        }
        container_t<T> add(const T& scalar) const {
            return map_to_new<T>([scalar](const T& x) -> T { return x + scalar; });
        }
        container_t<T> subtract(const container_t<T>& obj) const {
            return map_to_new<T>([](const T& x1, const T& x2) -> T { return x1 - x2; }, obj);
        }
        container_t<T> subtract(const T& scalar) const {
            return map_to_new<T>([scalar](const T& x) -> T { return x - scalar; });
        }
        container_t<T> multiply(const container_t<T>& obj) const {
            return map_to_new<T>([](const T& x1, const T& x2) -> T { return x1 * x2; }, obj);
        }
        container_t<T> multiply(const T& scalar) const {
            return map_to_new<T>([scalar](const T& x) -> T { return x * scalar; });
        }
        container_t<T> divide(const container_t<T>& obj) const {
            return map_to_new<T>([](const T& x1, const T& x2) -> T { return x1 / x2; }, obj);
        }
        container_t<T> divide(const T& scalar) const {
            return map_to_new<T>([scalar](const T& x) -> T { return x / scalar; });
        }
        container_t<T> square() const {
            return map_to_new<T>([](const T& x1) -> T { return x1 * x1; });
        }
        container_t<T> sqrt() const {
            return map_to_new<T>([](const T& x1) -> T { return T(std::sqrt(x1)); });
        }
        container_t<T> pow(T power) const {
            return map_to_new<T>([power](const T& x1) -> T { return T(std::pow(x1, power)); });
        }
        container_t<T> pow(const container_t<T>& powers) const {
            return map_to_new<T>([](const T& x1, const T& x2) -> T { return T(std::pow(x1, x2)); }, powers);
        }
        container_t<T> exp() const {
            return map_to_new<T>([](const T& x1) -> T { return std::exp(x1); });
        }
        container_t<T> exp2() const {
            return map_to_new<T>([](const T& x1) -> T { return std::exp2(x1); });
        }
        container_t<T> exp10() const {
            return map_to_new<T>([](const T& x1) -> T { return std::pow(T(10), x1); });
        }
        container_t<T> log() const {
            return map_to_new<T>([](const T& x1) -> T { return std::log(x1); });
        }
        container_t<T> log2() const {
            return map_to_new<T>([](const T& x1) -> T { return std::log2(x1); });
        }
        container_t<T> log10() const {
            return map_to_new<T>([](const T& x1) -> T { return std::log10(x1); });
        }
        container_t<T> sin() const {
            return map_to_new<T>([](const T& x1) -> T { return std::sin(x1); });
        }
        container_t<T> cos() const {
            return map_to_new<T>([](const T& x1) -> T { return std::cos(x1); });
        }
        container_t<T> tan() const {
            return map_to_new<T>([](const T& x1) -> T { return std::tan(x1); });
        }
        container_t<T> arcsin() const {
            return map_to_new<T>([](const T& x1) -> T { return std::asin(x1); });
        }
        container_t<T> arccos() const {
            return map_to_new<T>([](const T& x1) -> T { return std::acos(x1); });
        }
        container_t<T> arctan() const {
            return map_to_new<T>([](const T& x1) -> T { return std::atan(x1); });
        }
        container_t<T> deg2rad() const {
            return map_to_new<T>([](const T& x1) -> T { return x1 * T(PI / 180.0); });
        }
        container_t<T> rad2deg() const {
            return map_to_new<T>([](const T& x1) -> T { return x1 * T(180.0 / PI); });
        }
        container_t<T> abs() const {
            return map_to_new<T>([](const T& x1) -> T { return std::abs(x1); });
        }
        container_t<T> sign() const {
            return map_to_new<T>([](const T& x1) -> T { return (x1 > T(0)) - (x1 < T(0)); });
        }
        container_t<T> clamp(const T& min, const T& max) const {
            return map_to_new<T>([min, max](const T& x1) -> T { return std::clamp(x1, min, max); });
        }
        container_t<T> round() const {
            return map_to_new<T>([](const T& x1) -> T { return std::round(x1); });
        }
        container_t<T> floor() const {
            return map_to_new<T>([](const T& x1) -> T { return std::floor(x1); });
        }
        container_t<T> ceil() const {
            return map_to_new<T>([](const T& x1) -> T { return std::ceil(x1); });
        }
        container_t<T> trunc() const {
            return map_to_new<T>([](const T& x1) -> T { return std::trunc(x1); });
        }

        container_t<bool> operator<(const container_t<T>& obj) const {
            return less(obj);
        }
        container_t<bool> operator<(const T& scalar) const {
            return less(scalar);
        }
        container_t<bool> operator<=(const container_t<T>& obj) const {
            return less_equal(obj);
        }
        container_t<bool> operator<=(const T& scalar) const {
            return less_equal(scalar);
        }
        container_t<bool> operator>(const container_t<T>& obj) const {
            return greater(obj);
        }
        container_t<bool> operator>(const T& scalar) const {
            return greater(scalar);
        }
        container_t<bool> operator>=(const container_t<T>& obj) const {
            return greater_equal(obj);
        }
        container_t<bool> operator>=(const T& scalar) const {
            return greater_equal(scalar);
        }
        container_t<bool> operator==(const container_t<T>& obj) const {
            return equal(obj);
        }
        container_t<bool> operator==(const T& scalar) const {
            return equal(scalar);
        }
        container_t<bool> operator!=(const container_t<T>& obj) const {
            return not_equal(obj);
        }
        container_t<bool> operator!=(const T& scalar) const {
            return not_equal(scalar);
        }
        container_t<bool> operator||(const container_t<T>& obj) const {
            return logical_or(obj);
        }
        container_t<bool> operator||(const T& scalar) const {
            return logical_or(scalar);
        }
        container_t<bool> operator&&(const container_t<T>& obj) const {
            return logical_and(obj);
        }
        container_t<bool> operator&&(const T& scalar) const {
            return logical_and(scalar);
        }
        container_t<bool> operator!() const {
            return logical_not();
        }

        container_t<bool> less(const container_t<T>& obj) const {
            return map_to_new<bool>([](const T& x1, const T& x2) -> bool { return x1 < x2; }, obj);
        }
        container_t<bool> less(const T& scalar) const {
            return map_to_new<bool>([scalar](const T& x1) -> bool { return x1 < scalar; });
        }
        container_t<bool> less_equal(const container_t<T>& obj) const {
            return map_to_new<bool>([](const T& x1, const T& x2) -> bool { return x1 <= x2; }, obj);
        }
        container_t<bool> less_equal(const T& scalar) const {
            return map_to_new<bool>([scalar](const T& x1) -> bool { return x1 <= scalar; });
        }
        container_t<bool> greater(const container_t<T>& obj) const {
            return map_to_new<bool>([](const T& x1, const T& x2) -> bool { return x1 > x2; }, obj);
        }
        container_t<bool> greater(const T& scalar) const {
            return map_to_new<bool>([scalar](const T& x1) -> bool { return x1 > scalar; });
        }
        container_t<bool> greater_equal(const container_t<T>& obj) const {
            return map_to_new<bool>([](const T& x1, const T& x2) -> bool { return x1 >= x2; }, obj);
        }
        container_t<bool> greater_equal(const T& scalar) const {
            return map_to_new<bool>([scalar](const T& x1) -> bool { return x1 >= scalar; });
        }
        container_t<bool> equal(const container_t<T>& obj) const {
            return map_to_new<bool>([](const T& x1, const T& x2) -> bool { return x1 == x2; }, obj);
        }
        container_t<bool> equal(const T& scalar) const {
            return map_to_new<bool>([scalar](const T& x1) -> bool { return x1 == scalar; });
        }
        container_t<bool> not_equal(const container_t<T>& obj) const {
            return map_to_new<bool>([](const T& x1, const T& x2) -> bool { return x1 != x2; }, obj);
        }
        container_t<bool> not_equal(const T& scalar) const {
            return map_to_new<bool>([scalar](const T& x1) -> bool { return x1 != scalar; });
        }
        container_t<bool> logical_or(const container_t<T>& obj) const {
            return map_to_new<bool>([](const T& x1, const T& x2) -> bool { return x1 || x2; }, obj);
        }
        container_t<bool> logical_or(const T& scalar) const {
            return map_to_new<bool>([scalar](const T& x1) -> bool { return x1 || scalar; });
        }
        container_t<bool> logical_and(const container_t<T>& obj) const {
            return map_to_new<bool>([](const T& x1, const T& x2) -> bool { return x1 && x2; }, obj);
        }
        container_t<bool> logical_and(const T& scalar) const {
            return map_to_new<bool>([scalar](const T& x1) -> bool { return x1 && scalar; });
        }
        container_t<bool> logical_not() const {
            return map_to_new<bool>([](const T& x1) -> bool { return !x1; });
        }

        container_t<T> operator|(const container_t<T>& obj) const {
            return bitwise_or(obj);
        }
        container_t<T> operator|(const T& scalar) const {
            return bitwise_or(scalar);
        }
        container_t<T> operator&(const container_t<T>& obj) const {
            return bitwise_and(obj);
        }
        container_t<T> operator&(const T& scalar) const {
            return bitwise_and(scalar);
        }
        container_t<T> operator^(const container_t<T>& obj) const {
            return bitwise_xor(obj);
        }
        container_t<T> operator^(const T& scalar) const {
            return bitwise_xor(scalar);
        }
        container_t<T> operator~() const {
            return bitwise_not();
        }

        container_t<T> bitwise_or(const container_t<T>& obj) const {
            return map_to_new<T>([](const T& x1, const T& x2) -> T { return x1 | x2; }, obj);
        }
        container_t<T> bitwise_or(const T& scalar) const {
            return map_to_new<T>([scalar](const T& x1) -> T { return x1 | scalar; });
        }
        container_t<T> bitwise_and(const container_t<T>& obj) const {
            return map_to_new<T>([](const T& x1, const T& x2) -> T { return x1 & x2; }, obj);
        }
        container_t<T> bitwise_and(const T& scalar) const {
            return map_to_new<T>([scalar](const T& x1) -> T { return x1 & scalar; });
        }
        container_t<T> bitwise_xor(const container_t<T>& obj) const {
            return map_to_new<T>([](const T& x1, const T& x2) -> T { return x1 ^ x2; }, obj);
        }
        container_t<T> bitwise_xor(const T& scalar) const {
            return map_to_new<T>([scalar](const T& x1) -> T { return x1 ^ scalar; });
        }
        container_t<T> bitwise_not() const {
            return map_to_new<T>([](const T& x) -> T { return ~x; });
        }

	};
}
