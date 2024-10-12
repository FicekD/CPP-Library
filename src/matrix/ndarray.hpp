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

	template <typename Type, typename ObjType, typename ObjBool>
	class NDArray : public BaseArray<Type> {
    public:
        NDArray() noexcept : BaseArray<Type>() {}
        NDArray(std::size_t size) noexcept : BaseArray<Type>(size) {}

        static void _except_on_shape_mismatch(const Shape& shape1, const Shape& shape2) {
            if (shape1 != shape2)
                throw std::invalid_argument("Shape mismatch");
        }

        virtual Shape shape() const = 0;

        template <typename RetObjType = ObjType, typename RetType = Type>
        RetObjType map_to_new(const std::function<RetType(const Type&, const Type&)>& lambda, const ObjType& obj) const {
            NDArray::_except_on_shape_mismatch(this->shape(), obj.shape());
            RetObjType result(this->shape());
            BaseArray<Type>::template map_to<RetType>(lambda, dynamic_cast<const BaseArray<Type>&>(obj), dynamic_cast<BaseArray<RetType>&>(result));
            return result;
        }

        template <typename RetObjType = ObjType, typename RetType = Type>
        RetObjType map_to_new(const std::function<RetType(const Type&)>& lambda) const {
            RetObjType result(this->shape());
            BaseArray<Type>::template map_to<RetType>(lambda, dynamic_cast<BaseArray<RetType>&>(result));
            return result;
        }

#pragma region MATH_OPS

        ObjType operator+(const ObjType& obj) const {
            return add(obj);
        }
        ObjType operator+(const Type& scalar) const {
            return add(scalar);
        }
        ObjType operator+() const {
            return positive();
        }
        ObjType operator-(const ObjType& obj) const {
            return subtract(obj);
        }
        ObjType operator-(const Type& scalar) const {
            return subtract(scalar);
        }
        ObjType operator-() const {
            return negative();
        }
        ObjType operator*(const ObjType& obj) const {
            return multiply(obj);
        }
        ObjType operator*(const Type& scalar) const {
            return multiply(scalar);
        }
        ObjType operator/(const ObjType& obj) const {
            return divide(obj);
        }
        ObjType operator/(const Type& scalar) const {
            return divide(scalar);
        }
        ObjType positive() const {
            return map_to_new<ObjType, Type>([](const Type& x) -> Type { return +x; });
        }
        ObjType negative() const {
            return map_to_new<ObjType, Type>([](const Type& x) -> Type { return -x; });
        }
        ObjType add(const ObjType& obj) const {
            return map_to_new<ObjType, Type>([](const Type& x1, const Type& x2) -> Type { return x1 + x2; }, obj);
        }
        ObjType add(const Type& scalar) const {
            return map_to_new<ObjType, Type>([scalar](const Type& x) -> Type { return x + scalar; });
        }
        ObjType subtract(const ObjType& obj) const {
            return map_to_new<ObjType, Type>([](const Type& x1, const Type& x2) -> Type { return x1 - x2; }, obj);
        }
        ObjType subtract(const Type& scalar) const {
            return map_to_new<ObjType, Type>([scalar](const Type& x) -> Type { return x - scalar; });
        }
        ObjType multiply(const ObjType& obj) const {
            return map_to_new<ObjType, Type>([](const Type& x1, const Type& x2) -> Type { return x1 * x2; }, obj);
        }
        ObjType multiply(const Type& scalar) const {
            return map_to_new<ObjType, Type>([scalar](const Type& x) -> Type { return x * scalar; });
        }
        ObjType divide(const ObjType& obj) const {
            return map_to_new<ObjType, Type>([](const Type& x1, const Type& x2) -> Type { return x1 / x2; }, obj);
        }
        ObjType divide(const Type& scalar) const {
            return map_to_new<ObjType, Type>([scalar](const Type& x) -> Type { return x / scalar; });
        }
        ObjType square() const {
            return map_to_new<ObjType, Type>([](const Type& x1) -> Type { return x1 * x1; });
        }
        ObjType sqrt() const {
            return map_to_new<ObjType, Type>([](const Type& x1) -> Type { return Type(std::sqrt(x1)); });
        }
        ObjType pow(Type power) const {
            return map_to_new<ObjType, Type>([power](const Type& x1) -> Type { return Type(std::pow(x1, power)); });
        }
        ObjType pow(const ObjType& powers) const {
            return map_to_new<ObjType, Type>([](const Type& x1, const Type& x2) -> Type { return Type(std::pow(x1, x2)); }, powers);
        }
        ObjType exp() const {
            return map_to_new<ObjType, Type>([](const Type& x1) -> Type { return std::exp(x1); });
        }
        ObjType exp2() const {
            return map_to_new<ObjType, Type>([](const Type& x1) -> Type { return std::exp2(x1); });
        }
        ObjType exp10() const {
            return map_to_new<ObjType, Type>([](const Type& x1) -> Type { return std::pow(Type(10), x1); });
        }
        ObjType log() const {
            return map_to_new<ObjType, Type>([](const Type& x1) -> Type { return std::log(x1); });
        }
        ObjType log2() const {
            return map_to_new<ObjType, Type>([](const Type& x1) -> Type { return std::log2(x1); });
        }
        ObjType log10() const {
            return map_to_new<ObjType, Type>([](const Type& x1) -> Type { return std::log10(x1); });
        }
        ObjType sin() const {
            return map_to_new<ObjType, Type>([](const Type& x1) -> Type { return std::sin(x1); });
        }
        ObjType cos() const {
            return map_to_new<ObjType, Type>([](const Type& x1) -> Type { return std::cos(x1); });
        }
        ObjType tan() const {
            return map_to_new<ObjType, Type>([](const Type& x1) -> Type { return std::tan(x1); });
        }
        ObjType arcsin() const {
            return map_to_new<ObjType, Type>([](const Type& x1) -> Type { return std::asin(x1); });
        }
        ObjType arccos() const {
            return map_to_new<ObjType, Type>([](const Type& x1) -> Type { return std::acos(x1); });
        }
        ObjType arctan() const {
            return map_to_new<ObjType, Type>([](const Type& x1) -> Type { return std::atan(x1); });
        }
        ObjType deg2rad() const {
            return map_to_new<ObjType, Type>([](const Type& x1) -> Type { return x1 * Type(PI / 180.0); });
        }
        ObjType rad2deg() const {
            return map_to_new<ObjType, Type>([](const Type& x1) -> Type { return x1 * Type(180.0 / PI); });
        }
        ObjType abs() const {
            return map_to_new<ObjType, Type>([](const Type& x1) -> Type { return std::abs(x1); });
        }
        ObjType sign() const {
            return map_to_new<ObjType, Type>([](const Type& x1) -> Type { return (x1 > Type(0)) - (x1 < Type(0)); });
        }
        ObjType clamp(const Type& min, const Type& max) const {
            return map_to_new<ObjType, Type>([min, max](const Type& x1) -> Type { return std::clamp(x1, min, max); });
        }
        ObjType round() const {
            return map_to_new<ObjType, Type>([](const Type& x1) -> Type { return std::round(x1); });
        }
        ObjType floor() const {
            return map_to_new<ObjType, Type>([](const Type& x1) -> Type { return std::floor(x1); });
        }
        ObjType ceil() const {
            return map_to_new<ObjType, Type>([](const Type& x1) -> Type { return std::ceil(x1); });
        }
        ObjType trunc() const {
            return map_to_new<ObjType, Type>([](const Type& x1) -> Type { return std::trunc(x1); });
        }

#pragma endregion MATH_OPS

#pragma region LOGICAL_OPS

        ObjBool operator<(const ObjType& obj) const {
            return less(obj);
        }
        ObjBool operator<(const Type& scalar) const {
            return less(scalar);
        }
        ObjBool operator<=(const ObjType& obj) const {
            return less_equal(obj);
        }
        ObjBool operator<=(const Type& scalar) const {
            return less_equal(scalar);
        }
        ObjBool operator>(const ObjType& obj) const {
            return greater(obj);
        }
        ObjBool operator>(const Type& scalar) const {
            return greater(scalar);
        }
        ObjBool operator>=(const ObjType& obj) const {
            return greater_equal(obj);
        }
        ObjBool operator>=(const Type& scalar) const {
            return greater_equal(scalar);
        }
        ObjBool operator==(const ObjType& obj) const {
            return equal(obj);
        }
        ObjBool operator==(const Type& scalar) const {
            return equal(scalar);
        }
        ObjBool operator!=(const ObjType& obj) const {
            return not_equal(obj);
        }
        ObjBool operator!=(const Type& scalar) const {
            return not_equal(scalar);
        }
        ObjBool operator||(const ObjType& obj) const {
            return logical_or(obj);
        }
        ObjBool operator||(const Type& scalar) const {
            return logical_or(scalar);
        }
        ObjBool operator&&(const ObjType& obj) const {
            return logical_and(obj);
        }
        ObjBool operator&&(const Type& scalar) const {
            return logical_and(scalar);
        }
        ObjBool operator!() const {
            return logical_not();
        }

        ObjBool less(const ObjType& obj) const {
            return map_to_new<ObjBool, bool>([](const Type& x1, const Type& x2) -> bool { return x1 < x2; }, obj);
        }
        ObjBool less(const Type& scalar) const {
            return map_to_new<ObjBool, bool>([scalar](const Type& x1) -> bool { return x1 < scalar; });
        }
        ObjBool less_equal(const ObjType& obj) const {
            return map_to_new<ObjBool, bool>([](const Type& x1, const Type& x2) -> bool { return x1 <= x2; }, obj);
        }
        ObjBool less_equal(const Type& scalar) const {
            return map_to_new<ObjBool, bool>([scalar](const Type& x1) -> bool { return x1 <= scalar; });
        }
        ObjBool greater(const ObjType& obj) const {
            return map_to_new<ObjBool, bool>([](const Type& x1, const Type& x2) -> bool { return x1 > x2; }, obj);
        }
        ObjBool greater(const Type& scalar) const {
            return map_to_new<ObjBool, bool>([scalar](const Type& x1) -> bool { return x1 > scalar; });
        }
        ObjBool greater_equal(const ObjType& obj) const {
            return map_to_new<ObjBool, bool>([](const Type& x1, const Type& x2) -> bool { return x1 >= x2; }, obj);
        }
        ObjBool greater_equal(const Type& scalar) const {
            return map_to_new<ObjBool, bool>([scalar](const Type& x1) -> bool { return x1 >= scalar; });
        }
        ObjBool equal(const ObjType& obj) const {
            return map_to_new<ObjBool, bool>([](const Type& x1, const Type& x2) -> bool { return x1 == x2; }, obj);
        }
        ObjBool equal(const Type& scalar) const {
            return map_to_new<ObjBool, bool>([scalar](const Type& x1) -> bool { return x1 == scalar; });
        }
        ObjBool not_equal(const ObjType& obj) const {
            return map_to_new<ObjBool, bool>([](const Type& x1, const Type& x2) -> bool { return x1 != x2; }, obj);
        }
        ObjBool not_equal(const Type& scalar) const {
            return map_to_new<ObjBool, bool>([scalar](const Type& x1) -> bool { return x1 != scalar; });
        }
        ObjBool logical_or(const ObjType& obj) const {
            return map_to_new<ObjBool, bool>([](const Type& x1, const Type& x2) -> bool { return x1 || x2; }, obj);
        }
        ObjBool logical_or(const Type& scalar) const {
            return map_to_new<ObjBool, bool>([scalar](const Type& x1) -> bool { return x1 || scalar; });
        }
        ObjBool logical_and(const ObjType& obj) const {
            return map_to_new<ObjBool, bool>([](const Type& x1, const Type& x2) -> bool { return x1 && x2; }, obj);
        }
        ObjBool logical_and(const Type& scalar) const {
            return map_to_new<ObjBool, bool>([scalar](const Type& x1) -> bool { return x1 && scalar; });
        }
        ObjBool logical_not() const {
            return map_to_new<ObjBool, bool>([](const Type& x1) -> bool { return !x1; });
        }

#pragma endregion LOGICAL_OPS

#pragma region BITWISE_OPS

        ObjType operator|(const ObjType& obj) const {
            return bitwise_or(obj);
        }
        ObjType operator|(const Type& scalar) const {
            return bitwise_or(scalar);
        }
        ObjType operator&(const ObjType& obj) const {
            return bitwise_and(obj);
        }
        ObjType operator&(const Type& scalar) const {
            return bitwise_and(scalar);
        }
        ObjType operator^(const ObjType& obj) const {
            return bitwise_xor(obj);
        }
        ObjType operator^(const Type& scalar) const {
            return bitwise_xor(scalar);
        }
        ObjType operator~() const {
            return bitwise_not();
        }

        ObjType bitwise_or(const ObjType& obj) const {
            return map_to_new<ObjType, Type>([](const Type& x1, const Type& x2) -> Type { return x1 | x2; }, obj);
        }
        ObjType bitwise_or(const Type& scalar) const {
            return map_to_new<ObjType, Type>([scalar](const Type& x1) -> Type { return x1 | scalar; });
        }
        ObjType bitwise_and(const ObjType& obj) const {
            return map_to_new<ObjType, Type>([](const Type& x1, const Type& x2) -> Type { return x1 & x2; }, obj);
        }
        ObjType bitwise_and(const Type& scalar) const {
            return map_to_new<ObjType, Type>([scalar](const Type& x1) -> Type { return x1 & scalar; });
        }
        ObjType bitwise_xor(const ObjType& obj) const {
            return map_to_new<ObjType, Type>([](const Type& x1, const Type& x2) -> Type { return x1 ^ x2; }, obj);
        }
        ObjType bitwise_xor(const Type& scalar) const {
            return map_to_new<ObjType, Type>([scalar](const Type& x1) -> Type { return x1 ^ scalar; });
        }
        ObjType bitwise_not() const {
            return map_to_new<ObjType, Type>([](const Type& x) -> Type { return ~x; });
        }

#pragma endregion BITWISE_OPS

	};
}
