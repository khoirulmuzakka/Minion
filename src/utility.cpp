#include "utility.h"

namespace minion {

unsigned int global_seed = std::random_device{}();
static std::mt19937 rng(global_seed);

void set_global_seed(unsigned int seed) {
    global_seed = seed;
    rng.seed(global_seed);
}

std::mt19937& get_rng() {
    return rng;
}

double rand_gen(double low, double high) {
    std::uniform_real_distribution<> dis(low, high);
    return dis(get_rng());
}

std::vector<double> rand_gen(double low, double high, size_t N) {
    std::vector<double> samples(N);
    std::mt19937& gen = get_rng(); // Random number generator
    std::uniform_real_distribution<double> dis(low, high); // Uniform distribution over [low, high]

    for (size_t i = 0; i < N; ++i) {
        samples[i] = dis(gen);
    }

    return samples;
}

std::vector<size_t> random_choice(size_t Ninput, size_t N, bool replace) {
    if (N > Ninput && !replace) {
        throw std::invalid_argument("Cannot select more elements than are in the range without replacement");
    }

    // Generate initial vector of length Ninput
    std::vector<size_t> v(Ninput);
    std::iota(v.begin(), v.end(), 0); // Fill the vector with values 0 to Ninput - 1

    return random_choice(v, N, replace);
}

size_t rand_int(size_t n) {
    int n_ = static_cast<int> (n);
    std::uniform_int_distribution<> dis(0, n_ - 1);
    return dis(get_rng());
}

double rand_norm(double mu, double s) {
    std::mt19937& gen = get_rng();
    std::normal_distribution<double> dis(mu, s);
    return dis(gen);
}

double rand_cauchy(double location, double scale) {
    std::cauchy_distribution<double> distribution(location, scale);
    return distribution(get_rng());
}

std::vector<size_t> argsort(const std::vector<double>& v, bool ascending) {
    std::vector<size_t> indices(v.size());
    std::iota(indices.begin(), indices.end(), 0);

    if (ascending) {
        std::sort(indices.begin(), indices.end(), [&v](size_t i1, size_t i2) { return v[i1] < v[i2]; });
    } else { // Descending order
        std::sort(indices.begin(), indices.end(), [&v](size_t i1, size_t i2) { return v[i1] > v[i2]; });
    }
    return indices;
}

std::vector<std::vector<double>> latin_hypercube_sampling(const std::vector<std::pair<double, double>>& bounds, size_t population_size) {
    int dimensions = static_cast<int>(bounds.size());
    std::vector<std::vector<double>> sample(population_size, std::vector<double>(dimensions));
    std::uniform_real_distribution<double> dis(0.0, 1.0);

    for (int i = 0; i < dimensions; ++i) {
        std::vector<double> quantiles(population_size);
        for (int j = 0; j < population_size; ++j) {
            quantiles[j] = (j + dis(get_rng())) / population_size;
        }
        std::shuffle(quantiles.begin(), quantiles.end(), get_rng());
        for (int j = 0; j < population_size; ++j) {
            double lower_bound = bounds[i].first;
            double upper_bound = bounds[i].second;
            sample[j][i] = lower_bound + quantiles[j] * (upper_bound - lower_bound);
        }
    }
    return sample;
}

std::vector<std::vector<double>> random_sampling(const std::vector<std::pair<double, double>>& bounds, size_t population_size) {
    int dimensions = static_cast<int>(bounds.size());
    std::vector<std::vector<double>> sample;
    for (size_t i=0; i<population_size; i++) {
        std::vector<double> p;
        for (size_t j=0; j<dimensions; j++) p.push_back( rand_gen(bounds[j].first, bounds[j].second));
        sample.push_back(p);
    }
    return sample;
}

std::tuple<double, double> getMeanStd(const std::vector<double>& arr, const std::vector<double>& weight) {
    if (arr.size() != weight.size()) {
        throw std::invalid_argument("Arrays must have the same length.");
    }

    double sum_weights = std::accumulate(weight.begin(), weight.end(), 0.0);
    std::vector<double> normalized_weights(weight.size());
    for (size_t i = 0; i < weight.size(); ++i) {
        normalized_weights[i] = weight[i] / sum_weights;
    }

    double mean = 0.0;
    double variance = 0.0;
    for (size_t i = 0; i < arr.size(); ++i) {
        mean += arr[i] * normalized_weights[i];
    }
    for (size_t i = 0; i < arr.size(); ++i) {
        variance += normalized_weights[i] * std::pow(arr[i] - mean, 2);
    }
    double std_dev = std::sqrt(variance);

    return std::make_tuple(mean, std_dev);
}

double calcMean(const std::vector<double>& vec) {
    size_t len = vec.size() ;
    std::vector<double> weights (len, 1.0); 
    double mn, stand ;
    std::tie(mn, stand) = getMeanStd(vec, weights);
    return mn;
}

double calcStdDev(const std::vector<double>& vec) {
    size_t len = vec.size() ;
    std::vector<double> weights (len, 1.0); 
    double mn, stand ;
    std::tie(mn, stand) = getMeanStd(vec, weights);
    return stand;
}

void enforce_bounds(std::vector<std::vector<double>>& new_candidates, const std::vector<std::pair<double, double>>& bounds, const std::string& strategy) {
    int dim = static_cast<int>(bounds.size());
    std::random_device rd;
    std::mt19937 gen(rd());

    for (int d = 0; d < dim; ++d) {
        double lower_bound = bounds[d].first;
        double upper_bound = bounds[d].second;
        double e = upper_bound - lower_bound;  // Range of current dimension

        if (strategy == "clip") {
            // Clip values that exceed the bounds
            for (size_t i = 0; i < new_candidates.size(); ++i) {
                new_candidates[i][d] = clamp<double>(new_candidates[i][d], lower_bound, upper_bound);
            }
        } else if (strategy == "reflect") {
            // Reflect values back into bounds if they are out of range
            for (size_t i = 0; i < new_candidates.size(); ++i) {
                if (new_candidates[i][d] < lower_bound) {
                    new_candidates[i][d] = lower_bound + (lower_bound - new_candidates[i][d]);
                } else if (new_candidates[i][d] > upper_bound) {
                    new_candidates[i][d] = upper_bound - (new_candidates[i][d] - upper_bound);
                }
            }
        } else if (strategy == "random") {
            // Randomly sample a new value within the bounds for out-of-bounds candidates
            std::uniform_real_distribution<double> dis(lower_bound, upper_bound);
            for (size_t i = 0; i < new_candidates.size(); ++i) {
                if (new_candidates[i][d] < lower_bound || new_candidates[i][d] > upper_bound) {
                    new_candidates[i][d] = dis(gen);
                }
            }
        } else if (strategy == "reflect-random") {
            // Resample a new value within a limited range close to the bounds for out-of-range candidates
            for (size_t i = 0; i < new_candidates.size(); ++i) {
                if (new_candidates[i][d] < lower_bound) {
                    double d_lower = fabs(new_candidates[i][d] - lower_bound);
                    double low_range = lower_bound;
                    double high_range = lower_bound + std::min(d_lower, e);
                    std::uniform_real_distribution<double> dis(low_range, high_range);
                    new_candidates[i][d] = dis(gen);
                } else if (new_candidates[i][d] > upper_bound) {
                    double d_upper = fabs(new_candidates[i][d] - upper_bound);
                    double low_range = upper_bound - std::min(d_upper, e);
                    double high_range = upper_bound;
                    std::uniform_real_distribution<double> dis(low_range, high_range);
                    new_candidates[i][d] = dis(gen);
                }
            }
        } else if (strategy == "none") {}
        else {
            throw std::invalid_argument("Invalid strategy. Choose from 'clip', 'reflect', 'random', or 'random-leftover'.");
        }
    }
}

void printVectorOfVectors(const std::vector<std::vector<double>>& vec) {
    for (const auto& innerVec : vec) {
        for (const auto& value : innerVec) {
            std::cerr << value << " ";
        }
        std::cerr << std::endl;
    }
}

std::vector<double> normalize_vector(const std::vector<double>& input) {
    double sum = std::accumulate(input.begin(), input.end(), 0.0);
    std::vector<double> normalized(input.size());
    for (size_t i = 0; i < input.size(); ++i) {
        normalized[i] = input[i] / sum;
    };
    return normalized;
}

double euclideanDistance(const std::vector<double>& a, const std::vector<double>& b) {
        double sum = 0.0;
        for (size_t i = 0; i < a.size(); ++i) {
            sum += (a[i] - b[i]) * (a[i] - b[i]);
        }
        return std::sqrt(sum);
    }

// Function to calculate the center of particles
std::vector<double> calculateCenter(const std::vector<std::vector<double>>& particles) {
    size_t numParticles = particles.size();
    size_t dimension = particles[0].size();
    
    std::vector<double> center(dimension, 0.0);
    for (const auto& particle : particles) {
        for (size_t i = 0; i < dimension; ++i) {
            center[i] += particle[i];
        }
    }
    for (size_t i = 0; i < dimension; ++i) {
        center[i] /= numParticles;
    }
    return center;
}

// Function to calculate the average Euclidean distance to the center
double averageEuclideanDistance(const std::vector<std::vector<double>>& particles) {
    std::vector<double> center = calculateCenter(particles);
    double totalDistance = 0.0;
    for (const auto& particle : particles) {
        totalDistance += euclideanDistance(particle, center);
    }
    return totalDistance / particles.size();
} 

}