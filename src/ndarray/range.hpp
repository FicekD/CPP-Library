#ifndef _RANGE_H
#define _RANGE_H

#include <iostream>


namespace range {
	class Range {
        using range_type = std::size_t;

	public:
        range_type start = 0, stride = 1, stop = 0;

        Range() noexcept;
		Range(const Range&) noexcept;
		Range(Range&&) noexcept;
		Range(range_type start, range_type stride, range_type stop) noexcept;
		Range(range_type start, range_type stop) noexcept;

        void clear();
        std::size_t length() const;

        bool operator==(const Range& iter) const;

        class RangeIterator {
        private:
            Range* range_ptr = nullptr;
            std::size_t i = 0;
        public:
            using difference_type = std::ptrdiff_t;
            using element_type = range_type;
            using pointer = element_type*;
            using reference = element_type&;

            RangeIterator(Range* range_ptr);
            RangeIterator(Range* range_ptr, std::size_t i);

            RangeIterator& operator++();
            RangeIterator operator++(int);
            RangeIterator& operator--();
            RangeIterator operator--(int);

            bool operator==(const RangeIterator& iter) const;
            bool operator!=(const RangeIterator& iter) const;

            element_type operator*();
            const element_type operator*() const;
        };

        RangeIterator begin() const;
        RangeIterator end() const;
	};
}

#endif