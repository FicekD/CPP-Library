#include "random.hpp"

RandomGenerator::RandomGenerator() noexcept : 
	_generator(_random_device()), _normal_distribution(0.0, 1.0), _uniform_distribution(0.0, 1.0), _current_seed(0) {}

std::mt19937& RandomGenerator::generator() {
    return _generator;
}

std::normal_distribution<double> RandomGenerator::normal_distribution_sampler(double mean, double std) const {
    return std::normal_distribution<double>(mean, std);
}

std::uniform_real_distribution<double> RandomGenerator::uniform_distribution_sampler(double min, double max) const {
    return std::uniform_real_distribution<double>(min, max);
}

int RandomGenerator::seed() const {
    return _current_seed;
}

void RandomGenerator::seed(int new_seed) {
    _generator.seed(new_seed);
    _current_seed = new_seed;
}

double RandomGenerator::normal() {
    return _normal_distribution(_generator);
}

double RandomGenerator::normal(double mean, double std) {
    return normal() * std + mean;
}

double RandomGenerator::uniform() {
    return _uniform_distribution(_generator);
}

double RandomGenerator::uniform(double min, double max) {
    return uniform() * (max - min) + min;
}
