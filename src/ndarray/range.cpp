#include "range.hpp"

using namespace range;

Range::Range() noexcept {}

Range::Range(const Range& range) noexcept : start(range.start), stride(range.stride), stop(range.stop) {}

Range::Range(Range&& range) noexcept : start(range.start), stride(range.stride), stop(range.stop) {
	range.clear();
}

Range::Range(range_type start, range_type stride, range_type stop) noexcept : start(start), stride(stride), stop(stop) {}

Range::Range(range_type start, range_type stop) noexcept : start(start), stop(stop) {}

void Range::clear() {
	start = 0;
	stride = 1;
	stop = 0;
}

std::size_t Range::length() const {
	if (start == stop)
		return 0;
	return (stop - start - 1) / stride;
}

bool Range::operator==(const Range& iter) const {
	return start == iter.start && stride == iter.stride && stop == iter.stop;
}

Range::RangeIterator::RangeIterator(Range* range_ptr) : range_ptr(range_ptr), i(0) {}
Range::RangeIterator::RangeIterator(Range* range_ptr, std::size_t i) : range_ptr(range_ptr), i(i) {}

Range::RangeIterator& Range::RangeIterator::operator++() {
	++i;
	return *this;
}

Range::RangeIterator Range::RangeIterator::operator++(int) {
	Range::RangeIterator retval = *this;
	++(*this);
	return retval;
}

Range::RangeIterator& Range::RangeIterator::operator--() {
	--i;
	return *this;
}

Range::RangeIterator Range::RangeIterator::operator--(int) {
	Range::RangeIterator retval = *this;
	--(*this);
	return retval;
}

bool Range::RangeIterator::operator==(const RangeIterator& iter) const {
	return *(iter.range_ptr) == *(range_ptr) && iter.i == i;
}

bool Range::RangeIterator::operator!=(const RangeIterator& iter) const {
	return !(*this == iter);
}

Range::RangeIterator::element_type Range::RangeIterator::operator*() {
	return (range_ptr->start + i * range_ptr->stride);
}

Range::RangeIterator Range::begin() const {
	return Range::RangeIterator(const_cast<Range*>(this));
}

Range::RangeIterator Range::end() const {
	return Range::RangeIterator(const_cast<Range*>(this), length() + 1);
}
