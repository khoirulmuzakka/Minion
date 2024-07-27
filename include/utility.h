#ifndef UTILITY_H
#define UTILITY_H

#include <vector>
#include <random>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <tuple>
#include <functional>
#include <iostream>

extern unsigned int global_seed;

/**
 * @typedef MinionFunction
 * @brief A function type that takes a vector of vectors of doubles and a void pointer and returns a vector of doubles.
 */
typedef std::function<std::vector<double>(const std::vector<std::vector<double>>&, void*)> MinionFunction;

/**
 * @brief Set the global seed for the random number generator.
 * @param seed The seed to set.
 */
void set_global_seed(unsigned int seed);

/**
 * @brief Get the global random number generator.
 * @return A reference to the global random number generator.
 */
std::mt19937& get_rng();

/**
 * @brief Select a random subset of elements from a vector.
 * @tparam T The type of the elements in the vector.
 * @param v The input vector.
 * @param n The number of elements to select.
 * @param replace If true, selection is done with replacement.
 * @return A vector containing n randomly selected elements from the input vector.
 */
template <typename T>
std::vector<T> random_choice(const std::vector<T>& v, size_t n, bool replace = false) {
    std::vector<T> result;
    std::mt19937& rng = get_rng();

    if (replace) {
        std::uniform_int_distribution<size_t> dist(0, v.size() - 1);
        for (size_t i = 0; i < n; ++i) {
            result.push_back(v[dist(rng)]);
        }
    } else {
        std::vector<T> copy = v;
        std::shuffle(copy.begin(), copy.end(), rng);
        result.insert(result.end(), copy.begin(), copy.begin() + n);
    }

    return result;
}


/**
 * @brief Create a vector of length N with elements selected randomly from 0 to Ninput - 1.
 * @tparam T The type of the elements in the vector.
 * @param Ninput The length of the initial vector (0 to Ninput - 1).
 * @param N The number of elements to select.
 * @param replace If true, selection is done with replacement.
 * @return A vector of length N with randomly selected elements from 0 to Ninput - 1.
 */
std::vector<size_t> random_choice(size_t Ninput, size_t N, bool replace = false);


/**
 * @brief Normalize the elements of a vector.
 * @param input The input vector.
 * @return A vector with normalized elements.
 */
std::vector<double> normalize_vector(const std::vector<double>& input);

/**
 * @brief Generate a random double in the range [low, high).
 * @param low The lower bound of the range.
 * @param high The upper bound of the range.
 * @return A random double in the specified range.
 */
double rand_gen(double low=0.0, double high=1.0);

/**
 * @brief Generate random numbers within a specified range.
 *
 * Generates Nsample random numbers uniformly distributed within the range [low, high].
 *
 * @param low Lower bound of the range (inclusive).
 * @param high Upper bound of the range (inclusive).
 * @param N Number of random samples to generate.
 * @return Vector containing Nsample random numbers within the specified range.
 */
std::vector<double> rand_gen(double low, double high, size_t N);

/**
 * @brief Generate a random integer in the range [0, n-1].
 * @param n The upper bound of the range.
 * @return A random integer in the specified range.
 */
size_t rand_int(size_t n);

/**
 * @brief Sample from a normal distribution.
 *
 * Generates a random number from a normal distribution with specified mean
 * and standard deviation using the global random number generator.
 *
 * @param mu Mean of the normal distribution.
 * @param s Variance (or standard deviation) of the normal distribution.
 * @return Random sample from the normal distribution.
 */
double rand_norm(double mu, double s);

/**
 * @brief Get the indices that would sort a vector.
 * @param v The input vector.
 * @param ascending Whether to sort in ascending order (default is true).
 * @return A vector of indices that would sort the input vector.
 */
std::vector<size_t> argsort(const std::vector<double>& v, bool ascending = true);

/**
 * @brief Clamp a value between a low and high range.
 * @tparam T The type of the value.
 * @param value The value to clamp.
 * @param low The lower bound.
 * @param high The upper bound.
 * @return The clamped value.
 */
template <typename T>
T clamp(T value, T low, T high) {
    if (value < low) {
        return low;
    } else if (value > high) {
        return high;
    } else {
        return value;
    }
}

/**
 * @brief Print the elements of a vector to standard error.
 * @tparam T The type of the elements in the vector.
 * @param vec The input vector.
 */
template<typename T>
void printVector(const std::vector<T>& vec) {
    std::cerr << "[ ";
    for (const auto& elem : vec) {
        std::cerr << elem << " ";
    }
    std::cerr << "]" << std::endl;
}

/**
 * @brief Perform Latin Hypercube Sampling.
 * @param bounds The bounds for each dimension.
 * @param population_size The number of samples to generate.
 * @return A vector of vectors containing the samples.
 */
std::vector<std::vector<double>> latin_hypercube_sampling(const std::vector<std::pair<double, double>>& bounds, size_t population_size);

/**
 * @brief Calculate the mean and standard deviation of a vector.
 * @param arr The input vector.
 * @param weight The weights for each element.
 * @return A tuple containing the mean and standard deviation.
 */
std::tuple<double, double> getMeanStd(const std::vector<double>& arr, const std::vector<double>& weight);

/**
 * @brief Calculate the mean of a vector.
 * @param vec The input vector.
 * @return The mean of the vector.
 */
double calcMean(const std::vector<double>& vec);

/**
 * @brief Calculate the standard deviation of a vector.
 * @param vec The input vector.
 * @return The standard deviation of the vector.
 */
double calcStdDev(const std::vector<double>& vec);

/**
 * @brief Enforce bounds on a set of candidate solutions.
 * @param new_candidates The candidate solutions to enforce bounds on.
 * @param bounds The bounds for each dimension.
 * @param strategy The strategy for enforcing the bounds.
 */
void enforce_bounds(std::vector<std::vector<double>>& new_candidates, const std::vector<std::pair<double, double>>& bounds, const std::string& strategy);

/**
 * @brief Print a vector of vectors to standard error.
 * @param vec The input vector of vectors.
 */
void printVectorOfVectors(const std::vector<std::vector<double>>& vec);

#endif