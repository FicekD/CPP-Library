#ifndef _RANDOM_H
#define _RANDOM_H

#include <random>

class RandomGenerator {
private:
	std::random_device _random_device;
	std::mt19937 _generator;
	std::normal_distribution<double> _normal_distribution;
	std::uniform_real_distribution<double> _uniform_distribution;
	int _current_seed;
public:
	RandomGenerator() noexcept;

	std::mt19937& generator();

	std::normal_distribution<double> normal_distribution_sampler(double mean, double std) const;
	std::uniform_real_distribution<double> uniform_distribution_sampler(double min, double max) const;

	int seed() const;
	void seed(int seed);

	double normal();
	double normal(double mean, double std);

	double uniform();
	double uniform(double min, double max);
};

static RandomGenerator rng;

#endif